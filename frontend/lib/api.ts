import type { ClassResult } from './types';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

interface ClassifyResponse {
  classes: ClassResult[];
}

const getApiBaseUrl = (): string => {
  const configured = (import.meta.env as ImportMetaEnv & { VITE_API_URL?: string }).VITE_API_URL;
  if (configured && configured.trim().length > 0) {
    return configured.replace(/\/$/, '');
  }
  return '/api';
};

const API_BASE_URL = getApiBaseUrl();

const assertValidResponse = (payload: unknown): ClassifyResponse => {
  if (!payload || typeof payload !== 'object' || !('classes' in payload)) {
    throw new Error('Invalid API response: missing classes field.');
  }

  const maybeClasses = (payload as { classes: unknown }).classes;
  if (!Array.isArray(maybeClasses)) {
    throw new Error('Invalid API response: classes must be an array.');
  }

  return payload as ClassifyResponse;
};

export const classifyImage = async (imageDataUrl: string): Promise<ClassResult[]> => {
  const imageBlob = await (await fetch(imageDataUrl)).blob();
  const imageExt = imageBlob.type === 'image/png' ? 'png' : 'jpg';

  const formData = new FormData();
  formData.append('file', imageBlob, `lesion.${imageExt}`);

  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: 'POST',
    body: formData,
  });

  const json = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      json && typeof json === 'object' && 'detail' in json && typeof json.detail === 'string'
        ? json.detail
        : `Classification failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }

  const parsed = assertValidResponse(json);

  return [...parsed.classes].sort((a, b) => b.score - a.score);
};
