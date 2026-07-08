import type { PatientInfo } from '../../types/api';

interface PatientFormProps {
  value: PatientInfo;
  metadataLoaded?: boolean;
}

function ReadonlyField({
  label,
  value,
  placeholder
}: {
  label: string;
  value?: string;
  placeholder: string;
}) {
  return (
    <label className="text-sm font-medium text-slate-700">
      {label}
      <input
        className="input input-readonly mt-1"
        value={value || ''}
        placeholder={placeholder}
        readOnly
      />
    </label>
  );
}

export function PatientForm({ value, metadataLoaded = false }: PatientFormProps) {
  return (
    <div>
      {!metadataLoaded ? (
        <div className="mb-3 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          Chưa có dữ liệu bệnh nhân. Vui lòng upload metadata JSON hoặc QR do thiết bị/hệ thống sinh ra.
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <ReadonlyField
          label="Mã bệnh nhân"
          value={value.patient_code}
          placeholder="Tự động fill từ metadata"
        />
        <ReadonlyField
          label="Họ tên"
          value={value.patient_name}
          placeholder="Backend có thể tra từ mã bệnh nhân"
        />
        <ReadonlyField
          label="Liên hệ"
          value={value.contact}
          placeholder="Backend trả về nếu có"
        />
        <ReadonlyField
          label="Ghi chú"
          value={value.note}
          placeholder="Thông tin bổ sung"
        />
      </div>
    </div>
  );
}
