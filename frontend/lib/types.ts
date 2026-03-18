export interface ClassResult {
  id: string;
  name: string;
  score: number;
}

export interface AnalysisHistoryItem {
  id: string;
  date: string;
  time: string;
  classId: string;
  className: string;
  confidence: number;
  imageUrl?: string;
  imagePath?: string;
  allScores?: ClassResult[];
  notes?: string;
}
