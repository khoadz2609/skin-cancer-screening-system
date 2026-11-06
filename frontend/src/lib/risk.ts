import type { RiskLevel } from '../types/api';
export const SCREENING_THRESHOLD = 0.395996;
export const HIGH_RISK_THRESHOLD = 0.560058;

export function classifyRisk(probability: number, screeningThreshold = SCREENING_THRESHOLD, highRiskThreshold = HIGH_RISK_THRESHOLD): { risk_level: RiskLevel; risk_text: string } {
  if (probability >= highRiskThreshold) return { risk_level: 'high', risk_text: 'Nguy cơ cao / cần kiểm tra thêm' };
  if (probability >= screeningThreshold) return { risk_level: 'medium', risk_text: 'Cần kiểm tra thêm / không nên bỏ qua' };
  return { risk_level: 'low', risk_text: 'Nguy cơ thấp theo mô hình' };
}
export function riskBadgeClass(level: RiskLevel): string { return level === 'high' ? 'badge badge-high' : level === 'medium' ? 'badge badge-medium' : 'badge badge-low'; }
