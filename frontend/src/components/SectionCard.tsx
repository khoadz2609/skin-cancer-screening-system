import type { ReactNode } from 'react';
export function SectionCard({ title, subtitle, children, className = '' }: { title?: string; subtitle?: string; children: ReactNode; className?: string }) {
  return <section className={`card p-5 ${className}`}>{title ? <h2 className="text-base font-bold text-slate-900">{title}</h2> : null}{subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}<div className={title || subtitle ? 'mt-4' : ''}>{children}</div></section>;
}
