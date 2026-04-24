import type { ClassResult, InferenceMetadata, TrustUncertainty, TrustResult } from './types';
import { supabase } from './supabase';

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
  gradcam_image?: string | null; // Base64-encoded PNG heatmap overlay
  prediction: string;
  calibrated_confidence: number;
  uncertainty: TrustUncertainty;
  quality_flags: string[];
  recommendation: 'classify' | 'review_required' | 'reject';
}

export interface ClassifyResult {
  classes: ClassResult[];
  gradcamImage?: string; // Base64 data URL for display
  trustResult: TrustResult;
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

  if (!('recommendation' in payload)) {
    throw new Error('Invalid API response: missing recommendation field from Trust Layer.');
  }

  return payload as ClassifyResponse;
};

export const classifyImage = async (
  imageDataUrl: string,
  includeGradcam: boolean = true,
  metadata?: InferenceMetadata
): Promise<ClassifyResult> => {
  // Get current Supabase session
  // Supabase auto-refreshes tokens (autoRefreshToken: true in supabase.ts)
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session?.access_token) {
    throw new ApiError('Authentication required. Please log in again.', 401);
  }

  const imageBlob = await (await fetch(imageDataUrl)).blob();
  const imageExt = imageBlob.type === 'image/png' ? 'png' : 'jpg';

  const formData = new FormData();
  formData.append('file', imageBlob, `lesion.${imageExt}`);

  if (typeof metadata?.ageApprox === 'number' && Number.isFinite(metadata.ageApprox)) {
    formData.append('age_approx', metadata.ageApprox.toString());
  }

  const normalizedSex = metadata?.sex?.trim().toLowerCase();
  if (normalizedSex) {
    formData.append('sex', normalizedSex);
  }

  const normalizedAnatomSite = metadata?.anatomSite?.trim().toLowerCase();
  if (normalizedAnatomSite) {
    formData.append('anatom_site', normalizedAnatomSite);
  }

  // Add include_gradcam query parameter
  const url = new URL(`${API_BASE_URL}/classify`, window.location.origin);
  url.searchParams.set('include_gradcam', includeGradcam.toString());

  // Add a 30-second timeout to prevent indefinite hanging if the backend gets stuck
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  let response: Response;
  try {
    response = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${session.access_token}`,
      },
      body: formData,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('The analysis took too long and timed out. Please try a smaller image.', 408);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  const json = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      json && typeof json === 'object' && 'detail' in json && typeof json.detail === 'string'
        ? json.detail
        : `Classification failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }

  const parsed = assertValidResponse(json);
  const sortedClasses = [...parsed.classes].sort((a, b) => b.score - a.score);

  // Convert base64 to data URL if gradcam_image is present
  const gradcamImage = parsed.gradcam_image
    ? `data:image/png;base64,${parsed.gradcam_image}`
    : undefined;

  return {
    classes: sortedClasses,
    gradcamImage,
    trustResult: {
      prediction: parsed.prediction,
      calibrated_confidence: parsed.calibrated_confidence,
      uncertainty: parsed.uncertainty,
      quality_flags: parsed.quality_flags,
      recommendation: parsed.recommendation,
    },
  };
};
