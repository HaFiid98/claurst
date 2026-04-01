// OpenAI-compatible API client.
//
// Supports any server that implements the OpenAI /v1/chat/completions format:
//   - OpenAI          (https://api.openai.com)
//   - Ollama          (http://localhost:11434 by default)
//   - LM Studio       (http://localhost:1234 by default)
//   - Any other OpenAI-compatible endpoint
//
// This client converts Anthropic-style CreateMessageRequest / StreamEvent into
// OpenAI wire format on the way out, and converts OpenAI streaming chunks back
// into the same StreamEvent enum on the way in, so the rest of the codebase
// (cc_query, cc_tui, …) is completely unaware of the underlying provider.

use crate::client::ClientConfig;
use crate::streaming::{ContentDelta, StreamEvent, StreamHandler};
use crate::types::{ApiMessage, ApiToolDefinition, CreateMessageRequest, SystemPrompt};
use cc_core::error::ClaudeError;
use cc_core::types::{ContentBlock, UsageInfo};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// OpenAI wire types (request)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OAIRequest {
    model: String,
    messages: Vec<OAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OAIStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize, Clone)]
struct OAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct OAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OAIFunctionCall,
}

#[derive(Debug, Serialize, Clone)]
struct OAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OAIFunction,
}

#[derive(Debug, Serialize)]
struct OAIFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
struct OAIStreamOptions {
    include_usage: bool,
}

// ---------------------------------------------------------------------------
// OpenAI wire types (streaming response)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OAIChunk {
    #[serde(default)]
    #[allow(dead_code)]
    id: String,
    #[serde(default)]
    #[allow(dead_code)]
    model: String,
    choices: Vec<OAIChoice>,
    #[serde(default)]
    usage: Option<OAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OAIChoice {
    delta: OAIDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OAIDelta {
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OAIToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OAIFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct OAIFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

// ---------------------------------------------------------------------------
// Message format conversion  (Anthropic ApiMessage → OpenAI OAIMessage)
// ---------------------------------------------------------------------------

/// Convert a slice of Anthropic-style `ApiMessage`s into OpenAI-style messages.
///
/// Anthropic packs tool results as extra items inside the *user* turn's
/// content-block array. OpenAI expects them as separate messages with
/// role "tool".  The conversion also splits assistant messages that contain
/// both text *and* tool-use blocks because OpenAI expects a single
/// `tool_calls` field instead of interleaved blocks.
fn convert_messages(messages: &[ApiMessage]) -> Vec<OAIMessage> {
    let mut out: Vec<OAIMessage> = Vec::new();

    for msg in messages {
        let role = msg.role.as_str();

        match &msg.content {
            Value::String(text) => {
                out.push(OAIMessage {
                    role: role.to_string(),
                    content: Some(Value::String(text.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            Value::Array(blocks) => match role {
                "user" => convert_user_blocks(blocks, &mut out),
                "assistant" => convert_assistant_blocks(blocks, &mut out),
                _ => {
                    // Unknown role – pass through as-is.
                    out.push(OAIMessage {
                        role: role.to_string(),
                        content: Some(Value::Array(blocks.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            },

            other => {
                // Null or unexpected – skip.
                debug!("Skipping unexpected content value: {:?}", other);
            }
        }
    }

    out
}

/// Convert user-turn content blocks.
/// Text blocks → one "user" message.
/// tool_result blocks → individual "tool" messages.
fn convert_user_blocks(blocks: &[Value], out: &mut Vec<OAIMessage>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_call_id, content)

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(t.to_string());
                }
            }
            Some("tool_result") => {
                let id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let content = extract_tool_result_content(block);
                tool_results.push((id, content));
            }
            _ => {
                // image, document, etc. — best-effort: skip
                debug!("Skipping unsupported user content block type");
            }
        }
    }

    // If there's any text, emit a user message first.
    if !text_parts.is_empty() {
        out.push(OAIMessage {
            role: "user".to_string(),
            content: Some(Value::String(text_parts.join("\n"))),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Tool results each become a separate "tool" message.
    for (id, content) in tool_results {
        out.push(OAIMessage {
            role: "tool".to_string(),
            content: Some(Value::String(content)),
            tool_calls: None,
            tool_call_id: Some(id),
        });
    }
}

/// Convert assistant-turn content blocks.
/// Combines text blocks and tool_use blocks into one "assistant" message.
fn convert_assistant_blocks(blocks: &[Value], out: &mut Vec<OAIMessage>) {
    let mut text_content = String::new();
    let mut tool_calls: Vec<OAIToolCall> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                    text_content.push_str(t);
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block
                    .get("input")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                let arguments = serde_json::to_string(&input).unwrap_or_default();
                tool_calls.push(OAIToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: OAIFunctionCall { name, arguments },
                });
            }
            Some("thinking") | Some("redacted_thinking") => {
                // Thinking blocks are not sent to OpenAI models.
            }
            _ => {
                debug!("Skipping unsupported assistant content block type");
            }
        }
    }

    let content = if text_content.is_empty() {
        None
    } else {
        Some(Value::String(text_content))
    };
    let tool_calls_val = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    // Only push if there is something to say.
    if content.is_some() || tool_calls_val.is_some() {
        out.push(OAIMessage {
            role: "assistant".to_string(),
            content,
            tool_calls: tool_calls_val,
            tool_call_id: None,
        });
    }
}

/// Extract the text content from a tool_result block.
fn extract_tool_result_content(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => {
            // Fall back to the "output" field some callers use.
            block
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Tool definition conversion  (Anthropic → OpenAI)
// ---------------------------------------------------------------------------

fn convert_tools(tools: &[ApiToolDefinition]) -> Vec<OAITool> {
    tools
        .iter()
        .map(|t| OAITool {
            tool_type: "function".to_string(),
            function: OAIFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// OpenAI streaming → StreamEvent conversion state machine
// ---------------------------------------------------------------------------

/// Per-tool-call tracking accumulated while streaming.
struct ToolCallState {
    /// Our internal block index (different from OpenAI's tool_calls[N].index).
    block_index: usize,
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    name: String,
}

/// Stateful converter: feeds OpenAI SSE chunks and emits `StreamEvent`s.
struct OAIStreamConverter {
    /// Index counter for content blocks we emit.
    next_block_index: usize,
    /// Whether a text block is currently open.
    text_block_open: bool,
    /// Known tool calls indexed by OpenAI's tool_calls[N].index.
    tool_states: std::collections::HashMap<usize, ToolCallState>,
    /// Fake message id for MessageStart.
    message_id: String,
}

impl OAIStreamConverter {
    fn new() -> Self {
        Self {
            next_block_index: 0,
            text_block_open: false,
            tool_states: Default::default(),
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        }
    }

    fn emit_message_start(&self) -> StreamEvent {
        StreamEvent::MessageStart {
            id: self.message_id.clone(),
            model: String::new(),
            usage: UsageInfo::default(),
        }
    }

    /// Process one parsed chunk; push any resulting events into `events`.
    fn process_chunk(&mut self, chunk: OAIChunk, events: &mut Vec<StreamEvent>) {
        let choice = match chunk.choices.into_iter().next() {
            Some(c) => c,
            None => return,
        };

        let delta = choice.delta;
        let finish = choice.finish_reason;

        // ---- Text delta ----
        if let Some(text) = delta.content {
            if !text.is_empty() {
                if !self.text_block_open {
                    // Open a new text block.
                    events.push(StreamEvent::ContentBlockStart {
                        index: self.next_block_index,
                        content_block: ContentBlock::Text { text: String::new() },
                    });
                    self.text_block_open = true;
                    self.next_block_index += 1;
                }
                events.push(StreamEvent::ContentBlockDelta {
                    index: self.next_block_index - 1,
                    delta: ContentDelta::TextDelta { text },
                });
            }
        }

        // ---- Tool call deltas ----
        if let Some(tool_call_deltas) = delta.tool_calls {
            for tc_delta in tool_call_deltas {
                let oai_idx = tc_delta.index;

                if let Some(func) = &tc_delta.function {
                    // New tool call (has a name in the first chunk).
                    if let Some(name) = &func.name {
                        // Close text block if open.
                        if self.text_block_open {
                            events.push(StreamEvent::ContentBlockStop {
                                index: self.next_block_index - 1,
                            });
                            self.text_block_open = false;
                        }

                        let id = tc_delta
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", oai_idx));
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;

                        // Emit ContentBlockStart for this tool_use block.
                        events.push(StreamEvent::ContentBlockStart {
                            index: block_index,
                            content_block: ContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: Value::Object(Default::default()),
                            },
                        });

                        self.tool_states.insert(
                            oai_idx,
                            ToolCallState {
                                block_index,
                                id,
                                name: name.clone(),
                            },
                        );

                        // Emit any initial arguments fragment.
                        if let Some(args) = &func.arguments {
                            if !args.is_empty() {
                                events.push(StreamEvent::ContentBlockDelta {
                                    index: block_index,
                                    delta: ContentDelta::InputJsonDelta {
                                        partial_json: args.clone(),
                                    },
                                });
                            }
                        }
                    } else if let Some(args) = &func.arguments {
                        // Continuation delta (arguments fragment for an existing tool call).
                        if let Some(state) = self.tool_states.get(&oai_idx) {
                            if !args.is_empty() {
                                events.push(StreamEvent::ContentBlockDelta {
                                    index: state.block_index,
                                    delta: ContentDelta::InputJsonDelta {
                                        partial_json: args.clone(),
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }

        // ---- Finish ----
        if let Some(reason) = finish {
            // Close any open blocks.
            if self.text_block_open {
                events.push(StreamEvent::ContentBlockStop {
                    index: self.next_block_index - 1,
                });
                self.text_block_open = false;
            }
            for state in self.tool_states.values() {
                events.push(StreamEvent::ContentBlockStop {
                    index: state.block_index,
                });
            }
            self.tool_states.clear();

            // Translate OpenAI finish_reason → Anthropic stop_reason.
            let stop_reason = match reason.as_str() {
                "stop" => "end_turn",
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                other => other,
            };

            let usage = chunk.usage.map(|u| UsageInfo {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                ..Default::default()
            });

            events.push(StreamEvent::MessageDelta {
                stop_reason: Some(stop_reason.to_string()),
                usage,
            });
            events.push(StreamEvent::MessageStop);
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAICompatibleClient
// ---------------------------------------------------------------------------

/// An API client for any OpenAI-compatible endpoint.
pub struct OpenAICompatibleClient {
    http: reqwest::Client,
    config: ClientConfig,
}

impl OpenAICompatibleClient {
    /// Build a new client.  An empty `config.api_key` is allowed (e.g. Ollama
    /// running locally without authentication).
    pub fn new(config: ClientConfig) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .build()?;
        Ok(Self { http, config })
    }

    /// Non-streaming version: returns once the model has finished generating.
    pub async fn create_message(
        &self,
        request: CreateMessageRequest,
    ) -> Result<crate::types::CreateMessageResponse, ClaudeError> {
        // We run the streaming version and accumulate everything, which keeps
        // the conversion logic in one place.
        let handler: Arc<dyn StreamHandler> = Arc::new(crate::streaming::NullStreamHandler);
        let mut rx = self.create_message_stream(request, handler).await?;
        let mut acc = crate::StreamAccumulator::new();

        while let Some(evt) = rx.recv().await {
            acc.on_event(&evt);
            if matches!(evt, StreamEvent::MessageStop) {
                break;
            }
        }

        let (msg, usage, stop_reason) = acc.finish();
        let content = match &msg.content {
            cc_core::types::MessageContent::Text(t) => {
                vec![serde_json::json!({"type": "text", "text": t})]
            }
            cc_core::types::MessageContent::Blocks(blocks) => {
                serde_json::to_value(blocks)
                    .ok()
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default()
            }
        };

        Ok(crate::types::CreateMessageResponse {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content,
            model: self.config.api_base.clone(),
            stop_reason,
            stop_sequence: None,
            usage,
        })
    }

    /// Streaming version: sends the request and returns a channel of `StreamEvent`s.
    pub async fn create_message_stream(
        &self,
        request: CreateMessageRequest,
        handler: Arc<dyn StreamHandler>,
    ) -> Result<mpsc::Receiver<StreamEvent>, ClaudeError> {
        let oai_request = self.convert_request(request)?;
        let url = format!("{}/v1/chat/completions", self.config.api_base);

        let mut req = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        // Attach auth header (Bearer token) only when a key is provided.
        if !self.config.api_key.is_empty() {
            req = req.header(
                "Authorization",
                format!("Bearer {}", &self.config.api_key),
            );
        }

        let body = serde_json::to_value(&oai_request).map_err(ClaudeError::Json)?;
        let resp = req.json(&body).send().await.map_err(ClaudeError::Http)?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.map_err(ClaudeError::Http)?;
            return Err(ClaudeError::ApiStatus {
                status: status.as_u16(),
                message: text,
            });
        }

        let (tx, rx) = mpsc::channel(256);

        tokio::spawn(async move {
            if let Err(e) = Self::process_sse_stream(resp, handler, tx.clone()).await {
                let _ = tx
                    .send(StreamEvent::Error {
                        error_type: "stream_error".into(),
                        message: e.to_string(),
                    })
                    .await;
            }
        });

        Ok(rx)
    }

    // ---- Internal helpers -----------------------------------------------

    fn convert_request(
        &self,
        req: CreateMessageRequest,
    ) -> Result<OAIRequest, ClaudeError> {
        let mut messages: Vec<OAIMessage> = Vec::new();

        // System prompt → role "system" message.
        if let Some(system) = &req.system {
            let text = match system {
                SystemPrompt::Text(t) => t.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| b.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            messages.push(OAIMessage {
                role: "system".to_string(),
                content: Some(Value::String(text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Conversation history.
        messages.extend(convert_messages(&req.messages));

        // Tools.
        let (tools, tool_choice) = if let Some(ref tdefs) = req.tools {
            if tdefs.is_empty() {
                (None, None)
            } else {
                (
                    Some(convert_tools(tdefs)),
                    Some(Value::String("auto".to_string())),
                )
            }
        } else {
            (None, None)
        };

        Ok(OAIRequest {
            model: req.model,
            messages,
            tools,
            tool_choice,
            stream: true,
            stream_options: Some(OAIStreamOptions {
                include_usage: true,
            }),
            max_tokens: Some(req.max_tokens),
            temperature: req.temperature,
        })
    }

    async fn process_sse_stream(
        resp: reqwest::Response,
        handler: Arc<dyn StreamHandler>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> anyhow::Result<()> {
        use crate::sse_parser::SseLineParser;

        let mut parser = SseLineParser::new();
        let mut byte_stream = resp.bytes_stream();
        let mut leftover = String::new();

        let mut converter = OAIStreamConverter::new();

        // Emit MessageStart immediately so the TUI can start rendering.
        let start = converter.emit_message_start();
        handler.on_event(&start);
        if tx.send(start).await.is_err() {
            return Ok(());
        }

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result?;
            let text = String::from_utf8_lossy(&chunk);

            let combined = if leftover.is_empty() {
                text.to_string()
            } else {
                let mut s = std::mem::take(&mut leftover);
                s.push_str(&text);
                s
            };

            let mut lines: Vec<&str> = combined.split('\n').collect();
            if !combined.ends_with('\n') {
                leftover = lines.pop().unwrap_or("").to_string();
            }

            for line in lines {
                let line = line.trim_end_matches('\r');
                if let Some(frame) = parser.feed_line(line) {
                    // [DONE] marks end of stream.
                    if frame.data.trim() == "[DONE]" {
                        return Ok(());
                    }

                    match serde_json::from_str::<OAIChunk>(&frame.data) {
                        Ok(oai_chunk) => {
                            let mut events = Vec::new();
                            converter.process_chunk(oai_chunk, &mut events);
                            for evt in events {
                                handler.on_event(&evt);
                                if tx.send(evt).await.is_err() {
                                    return Ok(());
                                }
                            }
                        }
                        Err(e) => {
                            warn!(data = %frame.data, error = %e, "Failed to parse OAI chunk");
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
