import { riskBadgeClass } from '../../lib/risk';
import type { PredictionResponse } from '../../types/api';

interface PredictionResultProps {
  result: PredictionResponse | null;
}

export function PredictionResult({ result }: PredictionResultProps) {
  if (!result) {
    return (
      <div className="compact-result compact-result-empty">
        <span>Kết quả</span>
      </div>
    );
  }

  return (
    <div className="compact-result compact-result-done">
      <div>
        <p className="compact-result-label">Probability</p>
        <p className="compact-result-prob">{result.probability_malignant.toFixed(4)}</p>
      </div>

      <span className={riskBadgeClass(result.risk_level)}>{result.risk_level}</span>
    </div>
  );
}
