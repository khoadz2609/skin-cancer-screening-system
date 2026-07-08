import type { RiskLevel } from '../types/api';
export function classifyRisk(probability: number, screeningThreshold = 0.395996, highRiskThreshold = 0.560058): { risk_level: RiskLevel; risk_text: string } {
  if (probability >= highRiskThreshold) return { risk_level: 'high', risk_text: 'Nguy cơ cao / cần kiểm tra thêm' };
  if (probability >= screeningThreshold) return { risk_level: 'medium', risk_text: 'Cần kiểm tra thêm / không nên bỏ qua' };
  return { risk_level: 'low', risk_text: 'Nguy cơ thấp theo mô hình' };
}
export function riskBadgeClass(level: RiskLevel): string { return level === 'high' ? 'badge badge-high' : level === 'medium' ? 'badge badge-medium' : 'badge badge-low'; }
