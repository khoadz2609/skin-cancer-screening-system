export type RiskLevel = 'low' | 'medium' | 'high';

export interface PatientInfo {
  patient_code?: string;
  patient_name?: string;
  contact?: string;
  note?: string;
}

export interface MetadataPayload {
  [key: string]: unknown;

  metadata_mode?: 'raw' | 'processed' | 'lookup' | string;
  source?: string;

  patient_code?: string;
  patient_id?: string;
  isic_id?: string;
  metadata_ref?: string;

  patient?: PatientInfo;
  metadata?: Record<string, unknown>;
}

export interface MetadataResolveResponse {
  payload: MetadataPayload;
  patient?: PatientInfo;
  resolved_metadata?: MetadataPayload;
}

export interface HealthResponse {
  status: string;
  service?: string;
  model_ready: boolean;
  model_version?: string;
  threshold?: {
    screening_threshold: number;
    high_risk_threshold: number;
    target_recall?: number;
  };
}

export interface PredictionResponse {
  prediction_id?: string;
  model_version?: string;
  mode: string;
  probability_malignant: number;
  screening_threshold: number;
  high_risk_threshold: number;
  risk_level: RiskLevel;
  risk_text: string;
  recommendation?: string;
  metrics_context?: {
    holdout_auc?: number;
    holdout_average_precision?: number;
    holdout_recall_at_screening_threshold?: number;
    holdout_precision_at_screening_threshold?: number;
    holdout_false_negative?: number;
    holdout_false_positive?: number;
  };
  metadata_used?: MetadataPayload;
  patient?: PatientInfo;
  created_at?: string;
}

export interface PredictionHistoryItem {
  prediction_id: string;
  patient_code?: string;
  image_name: string;
  probability_malignant: number;
  risk_level: RiskLevel;
  risk_text: string;
  created_at: string;
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryItem[];
  total: number;
}

export interface BatchPredictionItem {
  isic_id: string;
  patient_id?: string;
  age_approx?: number;
  sex?: string;
  anatom_site_general?: string;
  clin_size_long_diam_mm?: number;
  probability_raw: number;
  probability_processed: number;
  risk_level: RiskLevel;
  risk_text: string;
  image_url: string;
  cam_image_url?: string;
}

export interface BatchPredictionResponse {
  status: string;
  results: BatchPredictionItem[];
  session_id: string;
}