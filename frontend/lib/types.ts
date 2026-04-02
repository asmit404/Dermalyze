export interface ClassResult {
  id: string;
  name: string;
  score: number;
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
  allScores?: ClassResult[];
  notes?: string;
}
