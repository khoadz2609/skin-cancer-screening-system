import { useEffect, useState } from 'react';
import { Activity, ClipboardList, ShieldCheck } from 'lucide-react';
import { getHealth } from './api/predictionApi';
import { env } from './config/env';
import { HistoryPage } from './features/history/HistoryPage';
import { PredictionPage } from './features/prediction/PredictionPage';
import type { HealthResponse } from './types/api';

type TabKey = 'predict' | 'history';

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('predict');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({
          status: 'error',
          model_ready: false
        })
      );
  }, []);

  const tabs = [
    { key: 'predict' as const, label: 'Dự đoán', icon: Activity },
    { key: 'history' as const, label: 'Lịch sử', icon: ClipboardList }
  ];

  return (
    <main className="app-shell">
      <div className="app-main">
        <header className="app-header flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-primary-600 p-3 text-white">
              <ShieldCheck size={26} />
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tight text-slate-950 md:text-3xl">
                Hệ thống hỗ trợ sàng lọc ung thư da
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Tích hợp AI dự đoán tỉ lệ ác tính dựa trên hình ảnh lâm sàng & Metadata
              </p>
            </div>
          </div>
        </header>

        <nav className="app-nav">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                className={`inline-flex items-center gap-2 rounded-2xl px-4 py-2 text-sm font-bold transition ${
                  isActive
                    ? 'bg-slate-950 text-white'
                    : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
                onClick={() => setActiveTab(tab.key)}
              >
                <Icon size={17} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        <section className="workspace">
          <div className={activeTab === 'predict' ? 'block' : 'hidden'}>
            <PredictionPage />
          </div>
          <div className={activeTab === 'history' ? 'block' : 'hidden'}>
            <HistoryPage isActive={activeTab === 'history'} />
          </div>
        </section>

        <footer className="text-center text-xs text-slate-500">
          Ứng dụng chỉ phục vụ demo học thuật, không dùng để chẩn đoán y khoa.
        </footer>
      </div>
    </main>
  );
}

export default App;
