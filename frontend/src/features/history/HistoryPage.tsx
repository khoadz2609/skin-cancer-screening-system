import { useEffect, useState } from 'react';
import { getPredictionHistory } from '../../api/predictionApi';
import { riskBadgeClass } from '../../lib/risk';
import type { PredictionHistoryItem } from '../../types/api';

export function HistoryPage({ isActive = true }: { isActive?: boolean }) {
  const [items, setItems] = useState<PredictionHistoryItem[]>([]);
  
  useEffect(() => {
    if (isActive) {
      getPredictionHistory().then((r) => setItems(r.items));
    }
  }, [isActive]);

  return (
    <div className="card p-5">
      <h2 className="text-lg font-bold text-slate-900">Lịch sử dự đoán</h2>
      <div className="mt-5 overflow-hidden rounded-2xl border">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-slate-500">
            <tr>
              <th className="px-4 py-3">Ảnh</th>
              <th className="px-4 py-3">Mã bệnh nhân</th>
              <th className="px-4 py-3">Probability</th>
              <th className="px-4 py-3">Risk</th>
              <th className="px-4 py-3">Thời gian</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.prediction_id} className="border-t bg-white">
                <td className="px-4 py-3 font-medium text-slate-800">{item.image_name}</td>
                <td className="px-4 py-3 text-slate-600">{item.patient_code || '-'}</td>
                <td className="px-4 py-3 font-bold">{item.probability_malignant.toFixed(4)}</td>
                <td className="px-4 py-3">
                  <span className={riskBadgeClass(item.risk_level)}>{item.risk_text}</span>
                </td>
                <td className="px-4 py-3 text-slate-500">
                  {new Date(item.created_at).toLocaleString('vi-VN')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
