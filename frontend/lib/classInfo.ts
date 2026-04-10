/**
 * Re-exports from classDefinitions.ts for backward compatibility.
 *
 * New code should import directly from './classDefinitions' instead.
 * This file is kept to avoid breaking existing imports.
 */

import {
  CLASS_DEFINITIONS,
  type RiskLevel,
  type RiskStyle,
  getRiskBadgeStyles as _getRiskBadgeStyles,
  getRiskLabel as _getRiskLabel,
  parseRiskLevel,
  getConfidenceColor as _getConfidenceColor,
} from './classDefinitions';

// Legacy interface for backward compatibility
export interface ClassInfo {
  id: string;
  name: string;
  description: string;
  riskLevel: string;
  commonIn: string;
  keyFeatures: string;
}

// Re-export RiskLevel as RiskSeverity for backward compat
export type RiskSeverity = RiskLevel;

// Build classInfoMap from CLASS_DEFINITIONS for backward compat
export const classInfoMap: Record<string, ClassInfo> = Object.fromEntries(
  Object.values(CLASS_DEFINITIONS).map((def) => [
    def.id,
    {
      id: def.id,
      name: def.name,
      description: def.description,
      riskLevel: def.riskLabel, // Use human-readable label
      commonIn: def.commonIn,
      keyFeatures: def.keyFeatures,
    },
  ])
);

// Re-export with original signatures
export const getRiskSeverity = parseRiskLevel;

export const getRiskBadgeStyles = (severity: RiskSeverity): RiskStyle =>
  _getRiskBadgeStyles(severity);

export const getRiskLabel = (severity: RiskSeverity): string => _getRiskLabel(severity);

export const getConfidenceColor = (confidence: number): string => _getConfidenceColor(confidence);
