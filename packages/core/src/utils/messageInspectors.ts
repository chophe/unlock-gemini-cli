/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content } from '../openai/openai-adapter.js';

export function isFunctionResponse(content: Content): boolean {
  return (
    content.role === 'user' &&
    !!content.parts &&
    content.parts.every((part) => !!part.functionResponse)
  );
}
