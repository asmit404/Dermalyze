export interface ClassResult {
  id: string;
  name: string;
  score: number;
}

export interface InferenceMetadata {
  ageApprox: number | null;
  sex: string | null;
  anatomSite: string | null;
}

export interface TrustUncertainty {
  score: number;
  normalized_entropy: number;
  top2_margin: number;
  variation_ratio: number;
}

export interface TrustResult {
  prediction: string;
  calibrated_confidence: number;
  uncertainty: TrustUncertainty;
  quality_flags: string[];
  recommendation: 'classify' | 'review_required' | 'reject';
}

export interface AnalysisHistoryItem {
  id: string;
  createdAt: string; // ISO timestamp for calculations
  date: string; // Formatted date for display
  time: string; // Formatted time for display
  classId: string;
  className: string;
  confidence: number;
  imageUrl?: string;
  imagePath?: string;
  gradcamUrl?: string; // Grad-CAM explainability heatmap
  gradcamPath?: string;
  allScores?: ClassResult[];
  notes?: string;
  metadata?: InferenceMetadata | null; // Patient metadata captured at analysis time
  imageDecryptFailed?: boolean; // True when the image exists but could not be decrypted
  trustRecommendation?: string;
  trustUncertaintyScore?: number;
  trustQualityFlags?: string[];
}

export interface DashStats {
  total: number;
  thisMonth: number;
  avgConfidence: number | null;
  needsReview: number;
  classCounts: { id: string; name: string; count: number }[];
  lastAnalysis: {
    className: string;
    classId: string;
    confidence: number;
    date: string;
    imageUrl: string | null;
  } | null;
}

// Re-export from single source of truth for backward compatibility
export { CLASS_NAMES as SHORT_CLASS_NAMES } from './classDefinitions';
