import { useState, useEffect } from 'react';
import { FileSpreadsheet, FileArchive, Loader2, ImageIcon, X, AlertCircle, Settings } from 'lucide-react';
import { toast } from 'sonner';

import { predictBatch, getModels } from '../../api/predictionApi';
import type { BatchPredictionResponse, BatchPredictionItem } from '../../types/api';
import { env } from '../../config/env';
import { riskBadgeClass } from '../../lib/risk';

export function PredictionPage() {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [mediaFile, setMediaFile] = useState<File | null>(null); // can be zip, hdf5, or single image
  const [modelFolder, setModelFolder] = useState<string>('models');
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BatchPredictionItem[] | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then((data) => {
        setAvailableModels(data.models);
        if (data.models.length > 0) {
          const hasModels = data.models.find(m => m.folder_name === 'models');
          if (hasModels) {
            setModelFolder('models');
          } else {
            setModelFolder(data.models[0].folder_name);
          }
        }
      })
      .catch((err) => {
        console.error('Error fetching models:', err);
        setAvailableModels([
          { folder_name: 'models', preset_name: 'kaggle_top1_fusion' },
          { folder_name: 'backbone_image', preset_name: 'image_only_train' }
        ]);
      });
  }, []);

  const isHdf5OrZip = (file: File | null) => {
    if (!file) return false;
    const name = file.name.toLowerCase();
    return name.endsWith('.zip') || name.endsWith('.h5') || name.endsWith('.hdf5');
  };

  const isImageFile = (file: File | null) => {
    if (!file) return false;
    const name = file.name.toLowerCase();
    return name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || name.endsWith('.webp');
  };

  const handlePredict = async () => {
    const isImageOnly = modelFolder === 'backbone_image';

    if (!isImageOnly && !csvFile) {
      toast.error('Vui lòng chọn file CSV chứa số liệu lâm sàng cho mô hình Kết hợp.');
      return;
    }

    if (!mediaFile) {
      toast.error('Vui lòng chọn Hình ảnh (ZIP/HDF5/Ảnh).');
      return;
    }
    
    setError(null);
    setIsLoading(true);
    setResults(null);
    
    try {
      const isZip = mediaFile && mediaFile.name.toLowerCase().endsWith('.zip');
      const isHdf5 = mediaFile && (mediaFile.name.toLowerCase().endsWith('.h5') || mediaFile.name.toLowerCase().endsWith('.hdf5'));
      const isImg = isImageFile(mediaFile);

      const response = await predictBatch({
        csvFile: csvFile || undefined,
        zipFile: isZip ? mediaFile : undefined,
        hdf5File: isHdf5 ? mediaFile : undefined,
        imageFiles: isImg && mediaFile ? [mediaFile] : undefined,
        modelFolder
      });
      setResults(response.results);
      toast.success('Dự đoán hoàn tất!');
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err.message || 'Lỗi khi gọi API';
      setError(msg);
      toast.error('Có lỗi xảy ra: ' + msg);
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (prob: number) => {
    if (prob >= 0.5) return 'text-red-600 bg-red-50';
    if (prob >= 0.17) return 'text-amber-600 bg-amber-50';
    return 'text-emerald-600 bg-emerald-50';
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="card p-6">
        <h2 className="mb-4 text-xl font-black text-slate-900">Nhập dữ liệu bệnh nhân</h2>

        {/* Model Selection Dropdown */}
        <div className="mb-6 rounded-2xl border border-slate-200 bg-white p-4">
          <label className="mb-2 block text-sm font-bold text-slate-700 flex items-center gap-2">
            <Settings size={16} className="text-primary-500" />
            <span>Cấu hình mô hình dự đoán</span>
          </label>
          <select
            value={modelFolder}
            onChange={(e) => {
              setModelFolder(e.target.value);
              setResults(null);
            }}
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-800 focus:border-primary-500 focus:bg-white focus:outline-none"
          >
            {availableModels.map((m) => (
              <option key={m.folder_name} value={m.folder_name}>
                {m.folder_name === 'backbone_image' 
                  ? 'Mô hình chỉ dùng ảnh (Vision Only - EfficientNet-B0)' 
                  : m.folder_name === 'models'
                  ? 'Mô hình Kết hợp (Multimodal Fusion: Image + Tabular)'
                  : `Mô hình: ${m.folder_name} (${m.preset_name})`}
              </option>
            ))}
          </select>
          <p className="mt-2 text-xs text-slate-500">
            {modelFolder === 'backbone_image' 
              ? '* Chế độ chỉ dùng ảnh: Chỉ yêu cầu tải lên hình ảnh lâm sàng, không cần file metadata CSV.' 
              : '* Chế độ kết hợp: Yêu cầu tải lên cả file metadata CSV và hình ảnh tương ứng để tối ưu độ chính xác.'}
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* CSV Upload */}
          <label className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 p-6 text-center transition hover:border-primary-400 hover:bg-primary-50 ${modelFolder === 'backbone_image' ? 'opacity-40 pointer-events-none' : ''}`}>
            <FileSpreadsheet className="mx-auto mb-3 text-slate-400 group-hover:text-primary-500" size={32} />
            <h3 className="font-bold text-slate-700">1. File Metadata (CSV)</h3>
            <p className="mb-4 text-xs text-slate-500">Chứa thông tin bệnh nhân tương ứng với ảnh lâm sàng</p>
            <div className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-primary-600 shadow-sm ring-1 ring-inset ring-slate-200 group-hover:bg-primary-600 group-hover:text-white group-hover:ring-primary-600 transition">
              Tải file lên
            </div>
            {csvFile && (
              <div className="mt-3 flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                <span>Đã chọn: {csvFile.name}</span>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    setCsvFile(null);
                    setResults(null);
                  }}
                  className="rounded-full p-0.5 hover:bg-emerald-200"
                >
                  <X size={14} />
                </button>
              </div>
            )}
            <input 
              type="file" 
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                setCsvFile(e.target.files?.[0] || null);
                setResults(null);
              }}
            />
          </label>

          {/* Media Upload */}
          <label className="group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 p-6 text-center transition hover:border-primary-400 hover:bg-primary-50">
            <FileArchive className="mx-auto mb-3 text-slate-400 group-hover:text-primary-500" size={32} />
            <h3 className="font-bold text-slate-700">2. Hình Ảnh (ZIP/HDF5/Ảnh)</h3>
            <p className="mb-4 text-xs text-slate-500">Chấp nhận file lẻ hoặc tải lên hàng loạt qua file ZIP</p>
            <div className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-primary-600 shadow-sm ring-1 ring-inset ring-slate-200 group-hover:bg-primary-600 group-hover:text-white group-hover:ring-primary-600 transition">
              Tải file lên
            </div>
            {mediaFile && (
              <div className="mt-3 flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                <span>Đã chọn: {mediaFile.name}</span>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    setMediaFile(null);
                    setResults(null);
                  }}
                  className="rounded-full p-0.5 hover:bg-emerald-200"
                >
                  <X size={14} />
                </button>
              </div>
            )}
            <input 
              type="file" 
              accept=".zip,.hdf5,.h5,.jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => {
                setMediaFile(e.target.files?.[0] || null);
                setResults(null);
              }}
            />
          </label>
        </div>

        <div className="mt-6 flex items-center justify-end border-t pt-6">
          <button
            onClick={handlePredict}
            disabled={
              isLoading ||
              (modelFolder !== 'backbone_image' && !csvFile) ||
              !mediaFile
            }
            className="inline-flex items-center gap-2 rounded-xl bg-primary-600 px-8 py-3 font-bold text-white transition hover:bg-primary-700 disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="animate-spin" size={18} /> : <ImageIcon size={18} />}
            Tiến hành Sàng lọc
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-xl bg-red-50 p-4 text-sm font-medium text-red-700">
            <AlertCircle size={18} />
            {error}
          </div>
        )}
      </div>

      {/* Results Table */}
      {results && (
        <div className="card p-6">
          <h2 className="mb-4 text-xl font-black text-slate-900">
            Kết quả ({results.length} bệnh nhân)
          </h2>
          
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 text-xs uppercase text-slate-700">
                <tr>
                  <th className="px-4 py-3 font-bold">Hình ảnh</th>
                  <th className="px-4 py-3 font-bold">ISIC ID</th>
                  <th className="px-4 py-3 font-bold">Thông tin Metadata</th>
                  <th className="px-4 py-3 font-bold">Heatmap (Grad-CAM)</th>
                  <th className="px-4 py-3 font-bold">Xác suất ác tính</th>
                  <th className="px-4 py-3 font-bold">Mức độ rủi ro</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {results
                  .sort((a, b) => b.probability_processed - a.probability_processed)
                  .map((r) => (
                  <tr key={r.isic_id} className="hover:bg-slate-50">
                    <td className="p-4">
                      <img 
                        src={env.apiBaseUrl + r.image_url} 
                        alt="Original" 
                        className="h-20 w-20 cursor-pointer rounded-xl object-cover shadow-sm transition hover:scale-110"
                        onClick={() => setSelectedImage(env.apiBaseUrl + r.image_url)}
                        onError={(e) => (e.currentTarget.src = '/placeholder.png')}
                      />
                    </td>
                    <td className="px-4 py-3 font-semibold text-slate-900">
                      {r.isic_id}
                    </td>
                    <td className="px-4 py-3 text-xs leading-relaxed">
                      <div><span className="font-semibold text-slate-500">Tuổi:</span> {r.age_approx || 'N/A'}</div>
                      <div><span className="font-semibold text-slate-500">Giới:</span> {r.sex || 'N/A'}</div>
                      <div><span className="font-semibold text-slate-500">Vị trí:</span> {r.anatom_site_general || 'N/A'}</div>
                    </td>
                    <td className="p-4">
                      {r.cam_image_url ? (
                        <img 
                          src={env.apiBaseUrl + r.cam_image_url} 
                          alt="Grad-CAM" 
                          className="h-20 w-20 cursor-pointer rounded-xl object-cover shadow-sm transition hover:scale-110"
                          onClick={() => setSelectedImage(env.apiBaseUrl + r.cam_image_url!)}
                        />
                      ) : (
                        <span className="text-xs italic text-slate-400">Không hỗ trợ</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className={`inline-flex items-center rounded-lg px-2.5 py-1 font-bold ${getRiskColor(r.probability_processed)}`}>
                        {(r.probability_processed * 100).toFixed(2)}%
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={riskBadgeClass(r.risk_level)}>{r.risk_text}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 text-right text-xs italic text-slate-500">
            * Lưu ý: Heatmap mang tính minh họa, không phản ánh đầy đủ vùng ảnh hưởng quyết định.
          </div>
        </div>
      )}

      {/* Image Zoom Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 p-6 backdrop-blur-sm"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative flex max-h-full w-full max-w-3xl flex-col items-center justify-center">
            <button 
              className="absolute -right-4 -top-4 z-10 rounded-full bg-white p-2 text-slate-900 shadow-xl hover:bg-slate-200 md:-right-12 md:top-0"
              onClick={() => setSelectedImage(null)}
            >
              <X size={24} />
            </button>
            <img 
              src={selectedImage} 
              alt="Zoomed" 
              className="max-h-[85vh] w-full rounded-2xl object-contain shadow-2xl" 
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  );
}