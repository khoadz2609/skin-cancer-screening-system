import { UploadCloud, X } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

interface FileDropzoneProps {
  file: File | null;
  accept: Record<string, string[]>;
  label: string;
  hint?: string;
  selectedHint?: string;
  onChange: (file: File | null) => void;
}

function formatFileSize(size: number): string {
  if (size < 1024) return `${size}B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}KB`;
  return `${(size / (1024 * 1024)).toFixed(1)}MB`;
}

export function FileDropzone({
  file,
  accept,
  label,
  hint,
  selectedHint,
  onChange
}: FileDropzoneProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept,
    maxFiles: 1,
    onDrop: (files) => onChange(files[0] || null)
  });

  return (
    <div
      {...getRootProps()}
      className={`relative cursor-pointer rounded-2xl border border-dashed p-4 transition ${
        isDragActive ? 'border-primary-500 bg-primary-50' : 'border-slate-300 bg-slate-50'
      }`}
    >
      <input {...getInputProps()} />

      <div className="flex items-center gap-3 pr-9">
        <div className="rounded-xl bg-white p-2 text-primary-600 shadow-sm">
          <UploadCloud size={22} />
        </div>

        <div className="min-w-0">
          <p className="truncate font-semibold text-slate-800">{file ? file.name : label}</p>
          <p className="truncate text-sm text-slate-500">
            {file
              ? selectedHint || `${formatFileSize(file.size)} · bấm để đổi file`
              : hint || 'Kéo thả hoặc bấm để chọn file'}
          </p>
        </div>
      </div>

      {file ? (
        <button
          type="button"
          className="absolute right-3 top-1/2 inline-flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 shadow-sm hover:bg-slate-50"
          onClick={(event) => {
            event.stopPropagation();
            onChange(null);
          }}
          title="Xóa file"
        >
          <X size={16} />
        </button>
      ) : null}
    </div>
  );
}
