import { env } from '../config/env';
import { classifyRisk } from '../lib/risk';
import type {
  BatchPredictionResponse,
  HealthResponse,
  MetadataPayload,
  MetadataResolveResponse,
  PatientInfo,
  PredictionHistoryResponse,
  PredictionResponse
} from '../types/api';
import { http } from './http';

function mockProbability(metadata: MetadataPayload): number {
  const isic = String(metadata?.isic_id || '');

  if (isic.includes('0015657')) return 0.201;
  if (isic.includes('0015729')) return 0.006;
  if (isic.includes('0015740')) return 0.529;

  if (isic.includes('8624259')) return 0.006;
  if (isic.includes('2138706')) return 0.201;
  if (isic.includes('5677120')) return 0.887;
  if (isic.includes('7324912')) return 0.963;
  if (isic.includes('2485456')) return 0.529;

  const age = Number(metadata?.age_approx || 55);
  const size = Number(metadata?.clin_size_long_diam_mm || 3);
  const base = Math.min(0.92, Math.max(0.04, 0.08 + age / 250 + size / 60));
  return Number(base.toFixed(4));
}

const MOCK_PATIENT: PatientInfo = {
  patient_code: 'IP_6074337',
  patient_name: 'Demo Patient 6074337',
  contact: '0900000000',
  note: 'Demo patient loaded from backend mock'
};

const MOCK_METADATA: MetadataPayload = {
  metadata_mode: 'processed',
  source: 'optical_device',
  isic_id: 'ISIC_0015657',
  patient_id: 'IP_6074337',
  age_approx: 45,
  sex: 'male',
  anatom_site_general: 'posterior torso',
  clin_size_long_diam_mm: 2.7,
  image_type: 'TBP tile: close-up',
  tbp_tile_type: '3D: XP',
  tbp_lv_areaMM2: 3.846876,
  tbp_lv_perimeterMM: 9.387248,
  tbp_lv_location: 'Torso Back Top Third',
  tbp_lv_location_simple: 'Torso Back',
  attribution: 'Memorial Sloan Kettering Cancer Center',
  copyright_license: 'CC-BY'
};

export async function getHealth(): Promise<HealthResponse> {
  if (env.enableMock) {
    return {
      status: 'ok',
      service: 'mock-ui',
      model_ready: true,
      model_version: 'mock-model',
      threshold: {
        screening_threshold: 0.1791,
        high_risk_threshold: 0.5,
        target_recall: 0.95
      }
    };
  }

  const res = await http.get<HealthResponse>('/health');
  return res.data;
}

export async function predictSkinCancer(input: {
  image: File;
  patient: PatientInfo;
  metadata: MetadataPayload;
  metadataMode?: 'raw' | 'processed' | 'lookup';
  source?: string;
}): Promise<PredictionResponse> {
  if (env.enableMock) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const probability = mockProbability(input.metadata);
    const risk = classifyRisk(probability, 0.1791, 0.5);

    return {
      prediction_id: `mock_${Date.now()}`,
      model_version: 'mock-ui-model',
      mode: 'mock',
      probability_malignant: probability,
      screening_threshold: 0.1791,
      high_risk_threshold: 0.5,
      ...risk,
      recommendation:
        risk.risk_level === 'low'
          ? 'Theo mô hình demo, mẫu này chưa vượt ngưỡng sàng lọc.'
          : 'Nên được chuyên gia/bác sĩ kiểm tra thêm.',
      metrics_context: {
        holdout_auc: 0.9163,
        holdout_average_precision: 0.7551,
        holdout_recall_at_screening_threshold: 0.9667,
        holdout_precision_at_screening_threshold: 0.2843,
        holdout_false_negative: 1,
        holdout_false_positive: 73
      },
      metadata_used: input.metadata,
      created_at: new Date().toISOString()
    };
  }

  const formData = new FormData();
  formData.append('image', input.image);
  formData.append('patient', JSON.stringify(input.patient));
  formData.append('metadata', JSON.stringify(input.metadata));

  if (input.metadataMode) {
    formData.append('metadata_mode', input.metadataMode);
  }

  if (input.source) {
    formData.append('source', input.source);
  }

  const res = await http.post<PredictionResponse>('/predictions', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  return res.data;
}

export async function decodeQr(qrImage: File): Promise<{
  payload: MetadataPayload;
  resolved_metadata?: MetadataPayload;
  patient?: PatientInfo;
}> {
  if (env.enableMock) {
    await new Promise((resolve) => setTimeout(resolve, 500));

    return {
      payload: {
        metadata_mode: 'lookup',
        source: 'device_or_hospital_system',
        patient_code: 'PT_DEMO_001',
        isic_id: 'ISIC_5677120',
        metadata_ref: 'META_20260627_0001'
      },
      patient: {
        patient_code: 'PT_DEMO_001',
        patient_name: 'Nguyen Van A',
        contact: '0900000000',
        note: 'Mock QR patient'
      },
      resolved_metadata: {
        metadata_mode: 'processed',
        source: 'mock_qr',
        isic_id: 'ISIC_5677120',
        age_approx: 60,
        sex: 'male',
        anatom_site_general: 'posterior torso',
        clin_size_long_diam_mm: 3.2
      }
    };
  }

  const formData = new FormData();
  formData.append('qr_image', qrImage);

  const res = await http.post<{
    payload: MetadataPayload;
    resolved_metadata?: MetadataPayload;
    patient?: PatientInfo;
  }>('/metadata/qr/decode', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  return res.data;
}

export async function resolveMetadataJson(file: File): Promise<MetadataResolveResponse> {
  if (env.enableMock) {
    await new Promise((resolve) => setTimeout(resolve, 500));

    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as MetadataPayload;

      const nestedMetadata =
        parsed.metadata && typeof parsed.metadata === 'object' && !Array.isArray(parsed.metadata)
          ? (parsed.metadata as MetadataPayload)
          : parsed;

      const patientCode =
        String(parsed.patient_code || parsed.patient_id || nestedMetadata.patient_id || 'IP_6074337');

      const isicId =
        String(parsed.isic_id || nestedMetadata.isic_id || 'ISIC_0015657');

      return {
        payload: {
          metadata_mode: String(parsed.metadata_mode || nestedMetadata.metadata_mode || 'processed'),
          source: String(parsed.source || nestedMetadata.source || 'optical_device'),
          patient_code: patientCode,
          isic_id: isicId
        },
        patient: {
          patient_code: patientCode,
          patient_name: `Demo Patient ${patientCode.replace('IP_', '')}`,
          contact: '0900000000',
          note: 'Demo patient loaded from backend mock'
        },
        resolved_metadata: {
          ...nestedMetadata,
          metadata_mode: String(parsed.metadata_mode || nestedMetadata.metadata_mode || 'processed'),
          source: String(parsed.source || nestedMetadata.source || 'optical_device'),
          isic_id: isicId,
          patient_id: String(nestedMetadata.patient_id || patientCode)
        }
      };
    } catch {
      return {
        payload: {
          metadata_mode: 'processed',
          source: 'optical_device',
          patient_code: 'IP_6074337',
          isic_id: 'ISIC_0015657'
        },
        patient: MOCK_PATIENT,
        resolved_metadata: MOCK_METADATA
      };
    }
  }

  const formData = new FormData();
  formData.append('metadata_file', file);

  const res = await http.post<MetadataResolveResponse>('/metadata/json/resolve', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  return res.data;
}

export async function lookupMetadata(isicId: string, mode: 'raw' | 'processed' = 'processed') {
  if (env.enableMock) {
    return {
      isic_id: isicId,
      mode,
      metadata: {
        metadata_mode: mode,
        source: mode,
        isic_id: isicId,
        age_approx: 55,
        sex: 'male',
        anatom_site_general: 'posterior torso',
        clin_size_long_diam_mm: 8.86
      }
    };
  }

  const res = await http.get(`/metadata/${encodeURIComponent(isicId)}`, {
    params: { mode }
  });

  return res.data;
}

export async function getPredictionHistory(): Promise<PredictionHistoryResponse> {
  if (env.enableMock) {
    await new Promise((resolve) => setTimeout(resolve, 400));

    return {
      items: [
        {
          prediction_id: 'pred_demo_001',
          patient_code: 'IP_6074337',
          image_name: 'ISIC_0015657.jpg',
          probability_malignant: 0.201,
          risk_level: 'medium',
          risk_text: 'Nguy cơ trung bình / nên kiểm tra thêm',
          created_at: new Date().toISOString()
        }
      ],
      total: 1
    };
  }

  const res = await http.get<PredictionHistoryResponse>('/predictions');
  return res.data;
}

export async function predictBatch(input: {
  csvFile?: File;
  imageFiles?: File[];
  zipFile?: File;
  hdf5File?: File;
  modelFolder: string;
}): Promise<BatchPredictionResponse> {
  const formData = new FormData();
  if (input.csvFile) {
    formData.append('csv_file', input.csvFile);
  }
  
  if (input.zipFile) {
    formData.append('zip_file', input.zipFile);
  }
  
  if (input.hdf5File) {
    formData.append('hdf5_file', input.hdf5File);
  }
  
  if (input.imageFiles && input.imageFiles.length > 0) {
    input.imageFiles.forEach(file => {
      formData.append('image_files', file);
    });
  }
  
  formData.append('model_folder', input.modelFolder);

  const res = await http.post<BatchPredictionResponse>('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  return res.data;
}

export async function getModels(): Promise<{ models: any[] }> {
  if (env.enableMock) {
    return {
      models: [
        {
          folder_name: 'models',
          preset_name: 'kaggle_top1_fusion',
          checkpoints: ['mid_fusion_lite_fold_0.pt'],
          cv_pauc: 0.175
        },
        {
          folder_name: 'backbone_image',
          preset_name: 'image_only_train',
          checkpoints: ['image_only_fold_0.pt'],
          cv_pauc: 0.146
        }
      ]
    };
  }

  const res = await http.get<{ models: any[] }>('/api/models');
  return res.data;
}