/**
 * Single Source of Truth for Skin Lesion Class Definitions
 *
 * All class-related data (names, descriptions, risk levels, colors) should be
 * imported from this file. This ensures consistency across the entire frontend
 * and makes it easy to update when the model changes.
 *
 * Future improvement: Fetch this data from a /metadata backend endpoint
 * to enable model updates without frontend deployments.
 */

// ============================================================================
// Types
// ============================================================================

export type ClassId = 'akiec' | 'bcc' | 'bkl' | 'df' | 'mel' | 'nv' | 'vasc';

export type RiskLevel = 'critical' | 'high' | 'moderate' | 'low';

export interface ClassDefinition {
  /** Short identifier (e.g., 'mel') */
  id: ClassId;
  /** Display name (e.g., 'Melanoma') */
  name: string;
  /** Full clinical name (e.g., 'Malignant Melanoma') */
  fullName: string;
  /** Clinical description for patient/clinician education */
  description: string;
  /** Risk categorization */
  riskLevel: RiskLevel;
  /** Human-readable risk label (e.g., 'Critical Risk') */
  riskLabel: string;
  /** Demographics most commonly affected */
  commonIn: string;
  /** Dermoscopic and clinical features */
  keyFeatures: string;
  /** Chart/badge color (hex) */
  color: string;
}

export interface RiskStyle {
  bg: string;
  text: string;
  border: string;
  dot: string;
}

// ============================================================================
// Class Definitions
// ============================================================================

export const CLASS_DEFINITIONS: Record<ClassId, ClassDefinition> = {
  mel: {
    id: 'mel',
    name: 'Melanoma',
    fullName: 'Malignant Melanoma',
    description:
      'An aggressive malignant tumor of melanocytes with significant metastatic potential if not detected early.',
    riskLevel: 'critical',
    riskLabel: 'Critical Risk',
    commonIn: 'Any adult population; risk increases with UV damage and atypical nevi history',
    keyFeatures:
      'Asymmetry, border irregularity, color variegation, diameter growth, evolution over time',
    color: '#f87171', // Soft rose
  },
  bcc: {
    id: 'bcc',
    name: 'Basal Cell Carcinoma',
    fullName: 'Basal Cell Carcinoma',
    description:
      'The most common skin cancer, usually slow-growing with low metastatic potential but can be locally destructive.',
    riskLevel: 'high',
    riskLabel: 'High Risk',
    commonIn: 'Adults with cumulative UV exposure, face and neck',
    keyFeatures: 'Pearly papule, rolled border, telangiectasia, possible central ulceration',
    color: '#fb923c', // Soft orange
  },
  akiec: {
    id: 'akiec',
    name: 'Actinic Keratosis',
    fullName: 'Actinic Keratoses / Intraepithelial Carcinoma',
    description:
      'A precancerous lesion caused by chronic sun exposure; may progress to squamous cell carcinoma if untreated.',
    riskLevel: 'moderate',
    riskLabel: 'Moderate Risk',
    commonIn: 'Older adults, fair skin, heavily sun-exposed areas',
    keyFeatures: 'Rough or scaly erythematous patches, persistent tenderness, superficial crusting',
    color: '#fbbf24', // Soft amber
  },
  bkl: {
    id: 'bkl',
    name: 'Benign Keratosis',
    fullName: 'Benign Keratosis-like Lesions',
    description:
      'A benign keratinocytic lesion group, often including seborrheic keratosis and lichenoid keratosis.',
    riskLevel: 'low',
    riskLabel: 'Low Risk',
    commonIn: 'Middle-aged and older adults',
    keyFeatures: 'Waxy or verrucous surface, well-circumscribed borders, variable pigmentation',
    color: '#34d399', // Soft emerald
  },
  df: {
    id: 'df',
    name: 'Dermatofibroma',
    fullName: 'Dermatofibroma',
    description:
      'A benign fibrohistiocytic skin nodule that is generally stable and non-malignant.',
    riskLevel: 'low',
    riskLabel: 'Low Risk',
    commonIn: 'Young to middle-aged adults, frequently lower extremities',
    keyFeatures: 'Firm papule or nodule, dimple sign, peripheral pigment network',
    color: '#22d3ee', // Soft cyan
  },
  nv: {
    id: 'nv',
    name: 'Melanocytic Nevus',
    fullName: 'Melanocytic Nevi (Mole)',
    description: 'A common benign melanocytic lesion that is typically stable over time.',
    riskLevel: 'low',
    riskLabel: 'Low Risk',
    commonIn: 'All age groups, often appearing in childhood and early adulthood',
    keyFeatures: 'Symmetric shape, regular borders, uniform pigmentation, stable appearance',
    color: '#a78bfa', // Soft violet
  },
  vasc: {
    id: 'vasc',
    name: 'Vascular Lesion',
    fullName: 'Vascular Lesions',
    description: 'A benign vascular proliferation such as angioma; usually non-cancerous.',
    riskLevel: 'low',
    riskLabel: 'Low Risk',
    commonIn: 'Adults, trunk and extremities',
    keyFeatures: 'Red to violaceous coloration, lacunar pattern, blanching in some cases',
    color: '#f472b6', // Soft pink
  },
};

// ============================================================================
// Derived Data (computed from CLASS_DEFINITIONS)
// ============================================================================

/** All class IDs in alphabetical order */
export const CLASS_IDS: ClassId[] = Object.keys(CLASS_DEFINITIONS).sort() as ClassId[];

/** Map of class ID to display name */
export const CLASS_NAMES: Record<ClassId, string> = Object.fromEntries(
  Object.values(CLASS_DEFINITIONS).map((c) => [c.id, c.name])
) as Record<ClassId, string>;

/** Map of class ID to chart color */
export const CLASS_COLORS: Record<ClassId, string> = Object.fromEntries(
  Object.values(CLASS_DEFINITIONS).map((c) => [c.id, c.color])
) as Record<ClassId, string>;

/** Classes grouped by risk level */
export const CLASSES_BY_RISK: Record<RiskLevel, ClassId[]> = {
  critical: CLASS_IDS.filter((id) => CLASS_DEFINITIONS[id].riskLevel === 'critical'),
  high: CLASS_IDS.filter((id) => CLASS_DEFINITIONS[id].riskLevel === 'high'),
  moderate: CLASS_IDS.filter((id) => CLASS_DEFINITIONS[id].riskLevel === 'moderate'),
  low: CLASS_IDS.filter((id) => CLASS_DEFINITIONS[id].riskLevel === 'low'),
};

/** High-risk class IDs (critical, high, or moderate) */
export const HIGH_RISK_CLASS_IDS: ClassId[] = [
  ...CLASSES_BY_RISK.critical,
  ...CLASSES_BY_RISK.high,
  ...CLASSES_BY_RISK.moderate,
];

// ============================================================================
// Risk Level Styling
// ============================================================================

export const RISK_COLORS: Record<RiskLevel, string> = {
  critical: '#dc2626', // red-600
  high: '#ea580c', // orange-600
  moderate: '#f59e0b', // amber-500
  low: '#10b981', // emerald-500
};

export const RISK_BADGE_STYLES: Record<RiskLevel, RiskStyle> = {
  critical: {
    bg: 'bg-red-50',
    text: 'text-red-700',
    border: 'border-red-200',
    dot: 'bg-red-500',
  },
  high: {
    bg: 'bg-orange-50',
    text: 'text-orange-700',
    border: 'border-orange-200',
    dot: 'bg-orange-500',
  },
  moderate: {
    bg: 'bg-amber-50',
    text: 'text-amber-700',
    border: 'border-amber-200',
    dot: 'bg-amber-500',
  },
  low: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-700',
    border: 'border-emerald-200',
    dot: 'bg-emerald-500',
  },
};

export const RISK_LABELS: Record<RiskLevel, string> = {
  critical: 'Critical Risk',
  high: 'High Risk',
  moderate: 'Moderate Risk',
  low: 'Low Risk',
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get class definition by ID. Returns undefined for unknown classes.
 */
export function getClassDefinition(classId: string): ClassDefinition | undefined {
  return CLASS_DEFINITIONS[classId as ClassId];
}

/**
 * Get class name by ID, with fallback for unknown classes.
 */
export function getClassName(classId: string): string {
  return CLASS_DEFINITIONS[classId as ClassId]?.name ?? classId;
}

/**
 * Get risk level for a class ID.
 */
export function getClassRiskLevel(classId: string): RiskLevel {
  return CLASS_DEFINITIONS[classId as ClassId]?.riskLevel ?? 'low';
}

/**
 * Get risk badge styles for a risk level.
 */
export function getRiskBadgeStyles(riskLevel: RiskLevel): RiskStyle {
  return RISK_BADGE_STYLES[riskLevel];
}

/**
 * Get risk label for display.
 */
export function getRiskLabel(riskLevel: RiskLevel): string {
  return RISK_LABELS[riskLevel];
}

/**
 * Parse a risk level string (e.g., "Moderate to High") to RiskLevel type.
 * Used for backward compatibility with existing data.
 */
export function parseRiskLevel(riskString: string): RiskLevel {
  const value = riskString.toLowerCase();
  if (value.includes('critical')) return 'critical';
  if (value.includes('high')) return 'high';
  if (value.includes('moderate')) return 'moderate';
  return 'low';
}

/**
 * Check if a class ID is considered high risk (critical, high, or moderate).
 */
export function isHighRiskClass(classId: string): boolean {
  return HIGH_RISK_CLASS_IDS.includes(classId as ClassId);
}

/**
 * Get confidence color based on percentage.
 */
export function getConfidenceColor(confidence: number): string {
  if (confidence >= 85) return 'text-emerald-600';
  if (confidence >= 70) return 'text-teal-600';
  if (confidence >= 50) return 'text-amber-600';
  return 'text-red-600';
}

/**
 * Validate if a string is a known class ID.
 */
export function isValidClassId(classId: string): classId is ClassId {
  return Object.prototype.hasOwnProperty.call(CLASS_DEFINITIONS, classId);
}
