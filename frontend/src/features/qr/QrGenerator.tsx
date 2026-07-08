import { QrCode } from 'lucide-react';

export function QrGenerator() {
  return (
    <div className="card h-full overflow-auto p-6">
      <div className="flex items-start gap-4">
        <div className="rounded-2xl bg-primary-600 p-3 text-white">
          <QrCode size={28} />
        </div>
        <div>
          <h2 className="text-xl font-black text-slate-900">Luồng QR metadata</h2>
          <p className="mt-2 max-w-3xl text-slate-600">
            Trong luồng thực tế, QR không được tạo thủ công trên giao diện này. Thiết bị chụp/quang học
            hoặc hệ thống bệnh viện sẽ sinh QR chứa mã tham chiếu hoặc metadata. Bác sĩ chỉ quét hoặc
            upload ảnh QR để hệ thống lấy metadata.
          </p>
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border bg-slate-50 p-5">
          <h3 className="font-black text-slate-900">Payload QR khuyến nghị</h3>
          <p className="mt-2 text-sm text-slate-600">
            QR nên chứa mã lookup thay vì nhét full metadata quá dài.
          </p>
          <pre className="mt-4 overflow-auto rounded-2xl bg-white p-4 text-xs text-slate-600">
{`{
  "metadata_mode": "lookup",
  "source": "device_or_hospital_system",
  "patient_code": "PT_DEMO_001",
  "isic_id": "ISIC_5677120",
  "metadata_ref": "META_20260627_0001"
}`}
          </pre>
        </div>

        <div className="rounded-2xl border bg-slate-50 p-5">
          <h3 className="font-black text-slate-900">Backend cần làm gì?</h3>
          <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-slate-600">
            <li>Nhận ảnh QR từ frontend hoặc nhận payload QR đã decode.</li>
            <li>Tra metadata theo <code>metadata_ref</code>, <code>isic_id</code> hoặc <code>patient_code</code>.</li>
            <li>Trả về patient info + metadata đầy đủ cho frontend.</li>
            <li>Khi predict, nhận ảnh gốc + metadata và trả kết quả risk.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
