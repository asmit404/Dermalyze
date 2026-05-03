import React from 'react';
import {
  ArrowPathIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  LifebuoyIcon,
  MagnifyingGlassPlusIcon,
  PhotoIcon,
} from '@heroicons/react/24/outline';

interface HelpScreenProps {
  onBack: () => void;
}

const QUICK_START = [
  {
    title: 'Capture a usable image',
    detail: 'Use a focused dermatoscopic JPEG/PNG with lesion centered and minimal glare.',
    Icon: PhotoIcon,
  },
  {
    title: 'Run and review',
    detail: 'Check top class, confidence, and heatmap before making follow-up decisions.',
    Icon: MagnifyingGlassPlusIcon,
  },
  {
    title: 'Record and export',
    detail: 'Add notes immediately and export PDF if you need to share the case.',
    Icon: ClockIcon,
  },
];

const TROUBLESHOOTING: Array<{
  issue: string;
  whyItHappens: string;
  whatToDo: string;
}> = [
  {
    issue: 'Upload rejected',
    whyItHappens: 'File is not JPEG/PNG or exceeds 10 MB.',
    whatToDo: 'Re-export as JPG/PNG and keep file size under 10 MB.',
  },
  {
    issue: 'Analysis rejected by Trust Layer',
    whyItHappens: 'Image quality is too low or out-of-distribution (OOD) for the AI model to safely process.',
    whatToDo: 'Ensure good lighting, centered lesion, and minimal hair/blur. Retake and re-upload the photo.',
  },
  {
    issue: 'Low-confidence result',
    whyItHappens: 'Image quality is limited or lesion presentation is atypical.',
    whatToDo: 'Retake with better focus/lighting and correlate with exam findings.',
  },
  {
    issue: 'History missing a new case',
    whyItHappens: 'Save may have failed due to connection interruption.',
    whatToDo: 'Re-run analysis or refresh History to confirm persistence.',
  },
];

const FAQ = [
  {
    question: 'What is the Model Trust System?',
    answer:
      'The Trust Layer is a safety mechanism that evaluates if an image is suitable for AI analysis. It rejects blurry, poorly lit, or out-of-distribution images to prevent unsafe or misleading diagnostic predictions.',
  },
  {
    question: 'How should confidence be interpreted?',
    answer:
      'Confidence reflects model certainty, not diagnostic certainty. Use it as one signal alongside history, exam, and follow-up plans.',
  },
  {
    question: 'When should I prioritize urgent follow-up?',
    answer:
      'Treat critical/high-risk predictions as immediate review triggers, especially when aligned with concerning clinical features.',
  },
  {
    question: 'What if heatmap focus looks clinically irrelevant?',
    answer:
      'Treat that as a caution flag. Re-capture the image and avoid over-weighting model output when visual rationale is inconsistent.',
  },
];

const HelpScreen: React.FC<HelpScreenProps> = ({ onBack }) => {
  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
      <section className="bg-gradient-to-br from-teal-600 to-teal-700 px-6 py-12 sm:py-16">
        <div className="max-w-6xl mx-auto">
          <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-white/20 backdrop-blur-sm text-white text-[11px] font-bold uppercase tracking-wider rounded-lg border border-white/30 mb-4">
            <LifebuoyIcon className="w-4 h-4" />
            Help Center
          </span>
          <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-4">
            How to get better results with Dermalyze
          </h1>
          <p className="text-teal-50 text-base leading-relaxed max-w-3xl mb-8">
            Practical guidance for image quality, result interpretation, and quick recovery when
            something goes wrong.
          </p>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white border-2 border-white/30 hover:bg-white/10 rounded-xl transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to Dashboard
          </button>
        </div>
      </section>

      <main className="flex-1">
        <section className="bg-slate-100 px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">
                Quick start in three actions
              </h2>
              <p className="text-sm text-slate-500">Use this checklist for every new case.</p>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              {QUICK_START.map(({ title, detail, Icon }) => (
                <article
                  key={title}
                  className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm"
                >
                  <div className="w-12 h-12 bg-teal-600 rounded-xl flex items-center justify-center mb-4 shadow-sm">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">{title}</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">{detail}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-white px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Troubleshooting</h2>
              <p className="text-sm text-slate-500">Most common issues and the fastest fix.</p>
            </div>
            <div className="space-y-4">
              {TROUBLESHOOTING.map(({ issue, whyItHappens, whatToDo }) => (
                <article
                  key={issue}
                  className="bg-slate-50 rounded-2xl border-2 border-slate-300 p-5"
                >
                  <div className="grid lg:grid-cols-[180px_1fr_1fr] gap-4">
                    <div className="flex items-center gap-2">
                      <ExclamationTriangleIcon className="w-5 h-5 text-amber-600 shrink-0" />
                      <p className="text-sm font-bold text-slate-900">{issue}</p>
                    </div>
                    <div>
                      <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                        Why it happens
                      </p>
                      <p className="text-sm text-slate-600 leading-relaxed">{whyItHappens}</p>
                    </div>
                    <div>
                      <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                        What to do now
                      </p>
                      <p className="text-sm text-slate-700 leading-relaxed font-medium">
                        {whatToDo}
                      </p>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-slate-50 px-6 py-12">
          <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-6">
            <article className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm">
              <h2 className="text-lg font-bold text-slate-900 mb-4">
                Result interpretation guardrails
              </h2>
              <div className="space-y-3">
                <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
                  <p className="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-2">
                    Use results for
                  </p>
                  <p className="text-sm text-emerald-800 leading-relaxed">
                    Triage support, documentation consistency, and longitudinal comparison of lesion
                    behavior.
                  </p>
                </div>
                <div className="bg-rose-50 border border-rose-200 rounded-xl p-4">
                  <p className="text-xs font-bold text-rose-700 uppercase tracking-wider mb-2">
                    Do not use results for
                  </p>
                  <p className="text-sm text-rose-800 leading-relaxed">
                    Standalone diagnosis, treatment decisions without clinical correlation, or
                    delaying indicated referral.
                  </p>
                </div>
              </div>
            </article>

            <article className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <ArrowPathIcon className="w-5 h-5 text-teal-600" />
                <h2 className="text-lg font-bold text-slate-900">Frequently asked questions</h2>
              </div>
              <div className="space-y-4">
                {FAQ.map(({ question, answer }) => (
                  <div
                    key={question}
                    className="pb-4 border-b border-slate-200 last:border-b-0 last:pb-0"
                  >
                    <p className="text-sm font-semibold text-slate-900 mb-1">{question}</p>
                    <p className="text-sm text-slate-600 leading-relaxed">{answer}</p>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>
      </main>

      <footer className="bg-white border-t-2 border-slate-300 py-8 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-xs text-slate-400 uppercase tracking-widest font-bold">
            Need reliable results? prioritize image quality first.
          </p>
          <p className="text-xs text-slate-400 mt-2">
            Dermalyze supports clinical judgment, not replaces it.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HelpScreen;
