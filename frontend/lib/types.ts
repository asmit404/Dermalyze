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
