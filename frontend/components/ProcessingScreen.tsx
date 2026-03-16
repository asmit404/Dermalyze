
import React, { useEffect, useRef, useState } from 'react';
import { classifyImage, ApiError } from '../lib/api';
import type { ClassResult } from '../lib/types';

interface ProcessingScreenProps {
  image: string | null;
  onComplete: (results: ClassResult[]) => void;
  onError: (message?: string, retryable?: boolean) => void;
}

const ProcessingScreen: React.FC<ProcessingScreenProps> = ({
  image,
  onComplete,
  onError,
}) => {
  const [statusText, setStatusText] = useState('Validating image type…');
  const calledRef = useRef(false); // prevent double-invoke in React StrictMode

  useEffect(() => {
    if (calledRef.current) return;
    calledRef.current = true;

    if (!image) {
      onError('No image was provided for classification.');
      return;
    }

    const run = async () => {
      try {
        setStatusText('Validating image type…');
        await new Promise((r) => setTimeout(r, 400));

        setStatusText('Running model inference…');
        const results = await classifyImage(image);

        setStatusText('Finalising results…');
        await new Promise((r) => setTimeout(r, 300));

        onComplete(results);
      } catch (err: unknown) {
        const msg =
          err instanceof Error
            ? err.message
            : 'An unexpected error occurred during classification.';
        const retryable = err instanceof ApiError && err.status === 503;
        onError(msg, retryable);
      }
    };

    run();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Render nothing while navigation away is pending — prevents the spinner
  // flashing on page refresh or direct URL access when image is not in state.
  if (!image) return null;

  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="max-w-md w-full text-center">
          <div className="bg-white rounded-3xl border border-slate-200 p-12 sm:p-16 shadow-sm flex flex-col items-center">

            <div className="relative mb-10">
              <div className="w-20 h-20 rounded-full border-4 border-slate-100" />
              <div className="absolute top-0 left-0 w-20 h-20 rounded-full border-4 border-teal-600 border-t-transparent animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-10 h-10 bg-teal-50 rounded-full animate-pulse flex items-center justify-center">
                  <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
            </div>

            <h2 className="text-2xl font-bold text-slate-900 mb-2 tracking-tight">
              Analyzing image…
            </h2>
            <p className="text-sm text-teal-600 font-medium mb-2">{statusText}</p>
            <p className="text-slate-400 text-xs mb-10 leading-relaxed px-4">
              Our AI-assisted system is evaluating the dermatoscopic features.
            </p>

            <div className="w-full space-y-3 opacity-40 pointer-events-none">
              <div className="w-full h-11 bg-slate-100 rounded-full" />
              <div className="w-full h-11 border border-slate-200 rounded-full" />
            </div>
          </div>
        </div>
      </main>

      <footer className="py-8 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Designed to assist medical professionals. Not a replacement for clinical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default ProcessingScreen;
