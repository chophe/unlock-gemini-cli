/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  DEFAULT_OPENAI_MODEL,
  DEFAULT_OPENAI_FLASH_MODEL,
} from '../config/models.js';

/**
 * Checks if the default "pro" model is rate-limited and returns a fallback "flash"
 * model if necessary. This function is designed to be silent.
 * @param apiKey The API key to use for the check.
 * @param currentConfiguredModel The model currently configured in settings.
 * @returns An object indicating the model to use, whether a switch occurred,
 *          and the original model if a switch happened.
 */
export async function getEffectiveModel(
  apiKey: string,
  currentConfiguredModel: string,
): Promise<string> {
  if (currentConfiguredModel !== DEFAULT_OPENAI_MODEL) {
    // Only check if the user is trying to use the specific pro model we want to fallback from.
    return currentConfiguredModel;
  }

  const modelToTest = DEFAULT_OPENAI_MODEL;
  const fallbackModel = DEFAULT_OPENAI_FLASH_MODEL;
  const baseURL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
  const endpoint = `${baseURL}/chat/completions`;
  
  const body = JSON.stringify({
    model: modelToTest,
    messages: [{ role: 'user', content: 'test' }],
    max_tokens: 1,
    temperature: 0,
  });

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 2000); // 2s timeout for the request

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.status === 429) {
      console.log(
        `[INFO] Your configured model (${modelToTest}) was temporarily unavailable. Switched to ${fallbackModel} for this session.`,
      );
      return fallbackModel;
    }
    // For any other case (success, other error codes), we stick to the original model.
    return currentConfiguredModel;
  } catch (_error) {
    clearTimeout(timeoutId);
    // On timeout or any other fetch error, stick to the original model.
    return currentConfiguredModel;
  }
}