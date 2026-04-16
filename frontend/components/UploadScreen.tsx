import React, { useState, useRef, useEffect } from 'react';
import Button from './ui/Button';
import type { InferenceMetadata } from '../lib/types';

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB
const MAX_DIMENSION_PX = 448; // resize longest edge to ≤ 448 px
const JPEG_QUALITY = 0.85;
const ANATOM_SITE_OPTIONS: Array<{ label: string; options: Array<{ value: string; label: string }> }> = [
  {
    label: 'Head & Neck',
    options: [
      { value: 'head/neck', label: 'Head / Neck' },
      { value: 'face', label: 'Face' },
      { value: 'scalp', label: 'Scalp' },
      { value: 'ear', label: 'Ear' },
    ],
  },
  {
    label: 'Torso',
    options: [
      { value: 'trunk', label: 'Trunk' },
      { value: 'chest', label: 'Chest' },
      { value: 'abdomen', label: 'Abdomen' },
      { value: 'back', label: 'Back' },
    ],
  },
  {
    label: 'Limbs',
    options: [
      { value: 'upper extremity', label: 'Upper Extremity' },
      { value: 'lower extremity', label: 'Lower Extremity' },
      { value: 'hand', label: 'Hand' },
      { value: 'foot', label: 'Foot' },
      { value: 'acral', label: 'Acral' },
    ],
  },
  {
    label: 'Other',
    options: [
      { value: 'genital', label: 'Genital' },
      { value: 'unknown', label: 'Unknown' },
    ],
  },
];
const ANATOM_SITE_VALUES = new Set(
  ANATOM_SITE_OPTIONS.flatMap((group) => group.options.map((option) => option.value))
);
const DROPDOWN_BASE_CLASS =
  'w-full appearance-none px-3.5 py-2.5 pr-10 rounded-lg border border-slate-300 bg-gradient-to-b from-white to-slate-50 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 text-sm shadow-sm transition-colors';

/** Compress/resize a data-URL image using canvas. PNGs are kept lossless; others use JPEG. */
function compressImage(dataUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, MAX_DIMENSION_PX / Math.max(img.width, img.height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(dataUrl);
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const isPng = dataUrl.startsWith('data:image/png');
      resolve(isPng ? canvas.toDataURL('image/png') : canvas.toDataURL('image/jpeg', JPEG_QUALITY));
    };
    img.onerror = () => reject(new Error('Failed to load image for compression.'));
    img.src = dataUrl;
  });
}

interface UploadScreenProps {
  selectedImage: string | null;
  metadata: InferenceMetadata;
  onImageSelect: (img: string | null) => void;
  onMetadataChange: (metadata: InferenceMetadata) => void;
  onBack: () => void;
  onRunClassification: () => void;
  onError: (message?: string) => void;
}

const UploadScreen: React.FC<UploadScreenProps> = ({
  selectedImage,
  metadata,
  onImageSelect,
  onMetadataChange,
  onBack,
  onRunClassification,
  onError,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const normalizedSiteValue = metadata.anatomSite?.trim().toLowerCase() ?? '';
  const selectedSiteValue = ANATOM_SITE_VALUES.has(normalizedSiteValue) ? normalizedSiteValue : '';

  // Always start with a clean slate — clear any image left over from a previous session
  useEffect(() => {
    onImageSelect(null);
    onMetadataChange({
      ageApprox: null,
      sex: null,
      anatomSite: null,
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value;
    if (rawValue.trim() === '') {
      onMetadataChange({ ...metadata, ageApprox: null });
      return;
    }

    const digitsOnly = rawValue.replace(/\D/g, '');
    if (digitsOnly.length === 0) {
      onMetadataChange({ ...metadata, ageApprox: null });
      return;
    }

    // Canonicalize values like 007 -> 7 while preserving a single 0.
    const normalizedDigits = digitsOnly.replace(/^0+(?=\d)/, '');
    const parsed = Number(normalizedDigits);
    if (Number.isNaN(parsed)) {
      return;
    }

    const clampedAge = Math.min(120, Math.max(0, parsed));
    onMetadataChange({ ...metadata, ageApprox: clampedAge });
  };

  const handleSexChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value.trim();
    onMetadataChange({ ...metadata, sex: value.length > 0 ? value : null });
  };

  const handleAnatomSiteChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value.trim().toLowerCase();
    onMetadataChange({ ...metadata, anatomSite: value.length > 0 ? value : null });
  };

  const validateAndProcessFile = (file: File) => {
    const validTypes = ['image/jpeg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      onError('Unsupported file type. Please upload a JPEG or PNG image.');
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      onError(
        `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum allowed size is 10 MB.`
      );
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const compressed = await compressImage(reader.result as string);
        onImageSelect(compressed);
      } catch {
        onError('Failed to process the image. Please try a different file.');
      }
    };
    reader.onerror = () => {
      onError('Could not read the file. Please try again.');
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const clearSelection = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelect(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="max-w-6xl w-full">
          <div className="bg-white rounded-3xl border border-slate-300 p-8 sm:p-12 shadow-sm">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2 tracking-tight">
                Upload Image for Analysis
              </h2>
              <p className="text-slate-500 text-sm">
                Please provide a clear dermatoscopic image of the lesion.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
              <div className="lg:col-span-7">
                <div
                  className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 flex flex-col items-center justify-center min-h-[300px] lg:min-h-[420px] p-6 text-center
                    ${selectedImage ? 'border-teal-500 bg-teal-50/10' : 'border-slate-300 hover:border-teal-400 hover:bg-slate-50/50'}
                    ${isDragging ? 'border-teal-500 bg-teal-50 ring-4 ring-teal-500/10' : ''}
                    ${!selectedImage ? 'cursor-pointer' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={selectedImage ? undefined : triggerFileInput}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="image/jpeg,image/png"
                    className="hidden"
                  />

                  {!selectedImage ? (
                    <>
                      <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-4">
                        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-semibold text-slate-700">
                          Click to upload or drag and drop
                        </p>
                        <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">
                          Accepted formats: JPG or PNG
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="w-full h-full flex flex-col items-center">
                      <div className="relative group max-w-full">
                        <img
                          src={selectedImage}
                          alt="Dermatoscopic Preview"
                          className="max-h-[300px] lg:max-h-[340px] w-auto rounded-lg shadow-md border border-white"
                        />
                        <button
                          onClick={clearSelection}
                          className="absolute -top-3 -right-3 bg-white text-slate-400 hover:text-red-500 rounded-full p-1.5 shadow-lg border border-slate-200 transition-colors"
                          title="Remove image"
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                        </button>
                      </div>
                      <p className="mt-4 text-xs font-medium text-teal-600 bg-teal-50 px-3 py-1 rounded-full">
                        Image selected successfully
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="lg:col-span-5">
                <div className="border border-slate-200 rounded-2xl p-4 sm:p-5 bg-slate-50/60">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-slate-700">Optional Metadata</h3>
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">
                      Improves prediction
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mb-4">
                    Add clinical context to send alongside the image for inference.
                  </p>

                  <div className="grid grid-cols-1 gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label htmlFor="upload-age-approx" className="text-xs font-semibold text-slate-600">
                        Age (years)
                      </label>
                      <input
                        id="upload-age-approx"
                        type="text"
                        inputMode="numeric"
                        pattern="[0-9]*"
                        autoComplete="off"
                        maxLength={3}
                        value={metadata.ageApprox === null ? '' : String(metadata.ageApprox)}
                        onChange={handleAgeChange}
                        placeholder="e.g. 54"
                        className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 text-sm text-slate-800 placeholder:text-slate-400"
                      />
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <label htmlFor="upload-sex" className="text-xs font-semibold text-slate-600">
                          Sex
                        </label>
                        {metadata.sex && (
                          <button
                            type="button"
                            onClick={() => onMetadataChange({ ...metadata, sex: null })}
                            className="text-[11px] font-medium text-slate-500 hover:text-teal-600 transition-colors"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <div className="relative">
                        <select
                          id="upload-sex"
                          value={metadata.sex ?? ''}
                          onChange={handleSexChange}
                          className={`${DROPDOWN_BASE_CLASS} ${metadata.sex ? 'text-slate-800' : 'text-slate-400'}`}
                        >
                          <option value="" disabled hidden></option>
                          <option value="female">Female</option>
                          <option value="male">Male</option>
                          <option value="unknown">Unknown</option>
                        </select>
                        <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-400">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 9l-7 7-7-7"
                            />
                          </svg>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="upload-anatom-site"
                          className="text-xs font-semibold text-slate-600"
                        >
                          Anatomical Site
                        </label>
                        {metadata.anatomSite && (
                          <button
                            type="button"
                            onClick={() => onMetadataChange({ ...metadata, anatomSite: null })}
                            className="text-[11px] font-medium text-slate-500 hover:text-teal-600 transition-colors"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <div className="relative">
                        <select
                          id="upload-anatom-site"
                          value={selectedSiteValue}
                          onChange={handleAnatomSiteChange}
                          className={`${DROPDOWN_BASE_CLASS} ${selectedSiteValue ? 'text-slate-800' : 'text-slate-400'}`}
                        >
                          <option value="" disabled hidden></option>
                          {ANATOM_SITE_OPTIONS.map((group) => (
                            <optgroup key={group.label} label={group.label}>
                              {group.options.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                        <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-400">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 9l-7 7-7-7"
                            />
                          </svg>
                        </div>
                      </div>
                      <p className="text-[11px] text-slate-400 px-0.5">
                        Choose the closest body region where the lesion appears.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-10 flex flex-col gap-3">
              <Button disabled={!selectedImage} onClick={onRunClassification}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  Run Classification
                </div>
              </Button>
              <Button variant="secondary" onClick={onBack}>
                Cancel
              </Button>
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

export default UploadScreen;
