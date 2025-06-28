/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';

// Type definitions to match the original GenAI interface
export interface Content {
  role: 'user' | 'model' | 'system' | 'assistant' | string;
  parts: Part[];
}

export interface Part {
  text?: string;
  functionCall?: FunctionCall;
  functionResponse?: FunctionResponse;
  inlineData?: {
    mimeType: string;
    data: string;
  };
  fileData?: {
    mimeType: string;
    fileUri: string;
  };
  thought?: string; // For thinking/reasoning parts
}

export interface FunctionCall {
  name: string;
  args: Record<string, unknown>;
  id?: string; // For tracking function calls
}

export interface FunctionResponse {
  name: string;
  response: Record<string, unknown>;
  id?: string; // For tracking function responses
}

export interface GenerateContentResponse {
  candidates?: Array<{
    content: Content;
    finishReason?: string;
    index?: number;
    safetyRatings?: Array<{
      category: string;
      probability: string;
    }>;
    urlContextMetadata?: any;
    groundingMetadata?: GroundingMetadata;
  }>;
  promptFeedback?: {
    blockReason?: string;
    safetyRatings?: Array<{
      category: string;
      probability: string;
    }>;
  };
  usageMetadata?: GenerateContentResponseUsageMetadata;
  automaticFunctionCallingHistory?: Content[];
  functionCalls?: FunctionCall[]; // For compatibility
  text?: string; // For compatibility
}

export interface GenerateContentResponseUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  
  // Additional properties for compatibility
  cachedContentTokenCount?: number;
  thoughtsTokenCount?: number;
  toolUsePromptTokenCount?: number;
}

export interface GenerateContentParameters {
  model: string;
  config?: GenerateContentConfig;
  contents: Content[];
}

export interface GenerateContentConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  stopSequences?: string[];
  systemInstruction?: Content | string;
  tools?: Tool[];
  responseMimeType?: string;
  responseSchema?: SchemaUnion;
  thinkingConfig?: ThinkingConfig;
  abortSignal?: AbortSignal;
  
  // Additional properties for compatibility
  cachedContent?: any;
  toolConfig?: ToolConfig;
  labels?: Record<string, string>;
  safetySettings?: SafetySetting[];
  candidateCount?: number;
  responseLogprobs?: boolean;
  logprobs?: boolean;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  routingConfig?: GenerationConfigRoutingConfig;
  modelSelectionConfig?: ModelSelectionConfig;
  responseModalities?: string[];
  mediaResolution?: MediaResolution;
  speechConfig?: SpeechConfigUnion;
  audioTimestamp?: boolean;
}

export interface CountTokensParameters {
  model: string;
  contents: Content[];
}

export interface CountTokensResponse {
  totalTokens?: number;
}

export interface EmbedContentParameters {
  model: string;
  contents: string[];
}

export interface EmbedContentResponse {
  embeddings: Array<{
    values: number[];
  }>;
}

export interface Tool {
  functionDeclarations: FunctionDeclaration[];
  urlContext?: any; // For web-fetch tool compatibility
  googleSearch?: any; // For web-search tool compatibility
}

export interface FunctionDeclaration {
  name: string;
  description: string;
  parameters?: SchemaUnion;
}

export interface SchemaUnion {
  type: string;
  properties?: Record<string, SchemaUnion>;
  items?: SchemaUnion;
  required?: string[];
  enum?: string[];
  description?: string;
  anyOf?: SchemaUnion[]; // For MCP compatibility
  default?: any; // For default values
  [key: string]: unknown; // Index signature for additional properties
}

export type PartListUnion = Part[];
export type PartUnion = Part;

// Additional types
export interface GroundingMetadata {
  webSearchQueries?: string[];
  searchEntryPoint?: {
    renderedContent?: string;
  };
  groundingChunks?: Array<{
    web?: {
      uri?: string;
      title?: string;
    };
  }>;
  groundingSupports?: Array<{
    segment?: {
      startIndex?: number;
      endIndex?: number;
    };
    groundingChunkIndices?: number[];
    confidenceScores?: number[];
  }>;
}

export enum Type {
  STRING = 'STRING',
  NUMBER = 'NUMBER',
  INTEGER = 'INTEGER',
  BOOLEAN = 'BOOLEAN',
  ARRAY = 'ARRAY',
  OBJECT = 'OBJECT'
}

export interface CallableTool {
  name: string;
  description: string;
  inputSchema: SchemaUnion;
  tool?: any; // For MCP tool compatibility
  callTool?: (params: any) => Promise<any>; // For MCP tool execution
}

// OpenAI Content Generator Implementation
export class OpenAIContentGenerator {
  private client: OpenAI;

  constructor(config: { apiKey?: string; baseURL?: string; httpOptions?: any }) {
    this.client = new OpenAI({
      apiKey: config.apiKey || process.env.OPENAI_API_KEY,
      baseURL: config.baseURL || process.env.OPENAI_BASE_URL,
      defaultHeaders: config.httpOptions?.headers,
    });
  }

  async generateContent(request: GenerateContentParameters): Promise<GenerateContentResponse> {
    try {
      const messages = this.convertContentsToMessages(request.contents);
      
      // Add system instruction if provided
      if (request.config?.systemInstruction) {
        let systemContent: string;
        if (typeof request.config.systemInstruction === 'string') {
          systemContent = request.config.systemInstruction;
        } else {
          systemContent = request.config.systemInstruction.parts
            .map(part => part.text)
            .filter(Boolean)
            .join('\n');
        }
        if (systemContent) {
          messages.unshift({ role: 'system', content: systemContent });
        }
      }

      const tools = request.config?.tools ? this.convertToolsToOpenAI(request.config.tools) : undefined;

      const completion = await this.client.chat.completions.create({
        model: request.model,
        messages,
        temperature: request.config?.temperature,
        top_p: request.config?.topP,
        max_tokens: request.config?.maxOutputTokens,
        stop: request.config?.stopSequences,
        tools,
        tool_choice: tools ? 'auto' : undefined,
      });

      return this.convertOpenAIResponseToGenAI(completion);
    } catch (error) {
      throw new Error(`OpenAI API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.generateContentStreamInternal(request);
  }

  private async *generateContentStreamInternal(request: GenerateContentParameters): AsyncGenerator<GenerateContentResponse> {
    try {
      const messages = this.convertContentsToMessages(request.contents);
      
      // Add system instruction if provided
      if (request.config?.systemInstruction) {
        let systemContent: string;
        if (typeof request.config.systemInstruction === 'string') {
          systemContent = request.config.systemInstruction;
        } else {
          systemContent = request.config.systemInstruction.parts
            .map(part => part.text)
            .filter(Boolean)
            .join('\n');
        }
        if (systemContent) {
          messages.unshift({ role: 'system', content: systemContent });
        }
      }

      const tools = request.config?.tools ? this.convertToolsToOpenAI(request.config.tools) : undefined;

      const stream = await this.client.chat.completions.create({
        model: request.model,
        messages,
        temperature: request.config?.temperature,
        top_p: request.config?.topP,
        max_tokens: request.config?.maxOutputTokens,
        stop: request.config?.stopSequences,
        tools,
        tool_choice: tools ? 'auto' : undefined,
        stream: true,
      });

      let accumulatedContent = '';
      let toolCalls: any[] = [];

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
        
        if (delta?.content) {
          accumulatedContent += delta.content;
          yield this.createStreamResponse(accumulatedContent, false);
        }

        if (delta?.tool_calls) {
          toolCalls.push(...delta.tool_calls);
        }

        if (chunk.choices[0]?.finish_reason) {
          yield this.createStreamResponse(accumulatedContent, true, toolCalls);
          break;
        }
      }
    } catch (error) {
      throw new Error(`OpenAI API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // OpenAI doesn't have a direct token counting API, so we'll estimate
    // This is a simplified implementation - in production you might want to use tiktoken
    const text = request.contents
      .flatMap(content => content.parts)
      .map(part => part.text || '')
      .join(' ');
    
    // Rough estimation: ~4 characters per token
    const estimatedTokens = Math.ceil(text.length / 4);
    
    return { totalTokens: estimatedTokens };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    try {
      const response = await this.client.embeddings.create({
        model: request.model.includes('embedding') ? request.model : 'text-embedding-3-small',
        input: request.contents,
      });

      return {
        embeddings: response.data.map(item => ({ values: item.embedding })),
      };
    } catch (error) {
      throw new Error(`OpenAI embeddings error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private convertContentsToMessages(contents: Content[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    
    contents.forEach(content => {
      // Ensure role is properly typed
      let role: 'user' | 'system' | 'assistant';
      if (content.role === 'model') {
        role = 'assistant';
      } else if (content.role === 'user' || content.role === 'system' || content.role === 'assistant') {
        role = content.role;
      } else {
        // Handle string roles that might come from legacy code
        role = content.role === 'assistant' ? 'assistant' : 'user';
      }
      
      if (content.parts.some(part => part.functionCall || part.functionResponse)) {
        // Handle function calls and responses
        const textParts = content.parts.filter(part => part.text);
        const functionCalls = content.parts.filter(part => part.functionCall);
        const functionResponses = content.parts.filter(part => part.functionResponse);

        if (functionCalls.length > 0) {
          messages.push({
            role: 'assistant',
            content: textParts.map(part => part.text).join('\n') || null,
            tool_calls: functionCalls.map(part => ({
              id: `call_${Math.random().toString(36).substr(2, 9)}`,
              type: 'function' as const,
              function: {
                name: part.functionCall!.name,
                arguments: JSON.stringify(part.functionCall!.args),
              },
            })),
          });
        }

        if (functionResponses.length > 0) {
          // For function responses, we need to create tool messages
          functionResponses.forEach(part => {
            messages.push({
              role: 'tool' as const,
              tool_call_id: `call_${Math.random().toString(36).substr(2, 9)}`,
              content: JSON.stringify(part.functionResponse!.response),
            });
          });
        }
      } else {
        const content_text = content.parts
          .map(part => {
            if (part.text) return part.text;
            if (part.inlineData) return `[${part.inlineData.mimeType} data]`;
            return '';
          })
          .filter(Boolean)
          .join('\n');

        messages.push({ role, content: content_text });
      }
    });
    
    return messages;
  }

  private convertToolsToOpenAI(tools: Tool[]): OpenAI.Chat.ChatCompletionTool[] {
    return tools.flatMap(tool =>
      tool.functionDeclarations.map(func => ({
        type: 'function' as const,
        function: {
          name: func.name,
          description: func.description,
          parameters: func.parameters ? this.convertSchemaToOpenAI(func.parameters) : undefined,
        },
      }))
    );
  }

  private convertSchemaToOpenAI(schema: SchemaUnion): any {
    const converted: any = {
      type: schema.type.toLowerCase(),
    };

    if (schema.description) {
      converted.description = schema.description;
    }

    if (schema.properties) {
      converted.properties = {};
      for (const [key, value] of Object.entries(schema.properties)) {
        converted.properties[key] = this.convertSchemaToOpenAI(value);
      }
    }

    if (schema.items) {
      converted.items = this.convertSchemaToOpenAI(schema.items);
    }

    if (schema.required) {
      converted.required = schema.required;
    }

    if (schema.enum) {
      converted.enum = schema.enum;
    }

    return converted;
  }

  private convertOpenAIResponseToGenAI(response: OpenAI.Chat.ChatCompletion): GenerateContentResponse {
    const choice = response.choices[0];
    if (!choice) {
      return { candidates: [] };
    }

    const parts: Part[] = [];
    
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if (toolCall.type === 'function') {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || '{}'),
            },
          });
        }
      }
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: choice.finish_reason || undefined,
          index: choice.index,
        },
      ],
      usageMetadata: {
        promptTokenCount: response.usage?.prompt_tokens,
        candidatesTokenCount: response.usage?.completion_tokens,
        totalTokenCount: response.usage?.total_tokens,
      },
    };
  }

  private createStreamResponse(content: string, isComplete: boolean, toolCalls?: any[]): GenerateContentResponse {
    const parts: Part[] = [];
    
    if (content) {
      parts.push({ text: content });
    }

    if (toolCalls && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        if (toolCall.type === 'function') {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || '{}'),
            },
          });
        }
      }
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: isComplete ? 'stop' : undefined,
          index: 0,
        },
      ],
    };
  }
}

// Export all types and the main class
export * from './openai-types.js';

export interface SendMessageParameters {
  message: Part | PartListUnion;
  config?: GenerateContentConfig; // For compatibility
}

export function createUserContent(message: Part | PartListUnion): Content {
  const parts = Array.isArray(message) ? message : [message];
  return {
    role: 'user',
    parts,
  };
}

// Helper function to convert string to Part array for backward compatibility
export function stringToParts(text: string): Part[] {
  return [{ text }];
}

// Helper function to convert string to PartListUnion
export function stringToPartListUnion(text: string): PartListUnion {
  return [{ text }];
}

export interface SpeechConfigUnion {
  voiceConfig?: {
    prebuiltVoiceConfig?: {
      voiceName?: string;
    };
  };
}

export interface ThinkingConfig {
  includeThoughts?: boolean;
  thinkingBudget?: number; // For compatibility
}

export interface ToolListUnion extends Array<Tool> {}

export interface ToolConfig {
  functionCallingConfig?: {
    mode?: string;
    allowedFunctionNames?: string[];
  };
}

export function mcpToTool(tool: any): CallableTool {
  // Simple conversion from MCP tool to CallableTool
  return {
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
  };
}

// Additional type aliases for compatibility
export type Schema = SchemaUnion;
export type ContentListUnion = Content[];
export type ContentUnion = Content;
export type GenerationConfigRoutingConfig = any;
export type MediaResolution = any;
export type Candidate = {
  content: Content;
  finishReason?: string;
  index?: number;
  safetyRatings?: Array<{
    category: string;
    probability: string;
  }>;
};
export type ModelSelectionConfig = any;
export type GenerateContentResponsePromptFeedback = any;
export type SafetySetting = any;

// Mock constructor for GenerateContentResponse (for compatibility)
export class GenerateContentResponse {
  candidates?: Array<{
    content: Content;
    finishReason?: string;
    index?: number;
    safetyRatings?: Array<{
      category: string;
      probability: string;
    }>;
    urlContextMetadata?: any;
    groundingMetadata?: GroundingMetadata;
  }>;
  promptFeedback?: {
    blockReason?: string;
    safetyRatings?: Array<{
      category: string;
      probability: string;
    }>;
  };
  usageMetadata?: GenerateContentResponseUsageMetadata;
  automaticFunctionCallingHistory?: Content[];
  functionCalls?: FunctionCall[];
  text?: string;

  constructor(data: any) {
    Object.assign(this, data);
  }
}