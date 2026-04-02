import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import type { ClassResult } from './types';
import type { ClassInfo, RiskSeverity } from './classInfo';
import { getRiskSeverity } from './classInfo';

export interface ReportData {
  caseId: string;
  date: string;
  time?: string;
  clinicianName: string;
  classId: string;
  className: string;
  confidence: number;
  classInfo: ClassInfo;
  allScores?: ClassResult[] | null;
  notes?: string;
  imageDataUrl?: string;
}

function getRiskColor(severity: RiskSeverity): string {
  switch (severity) {
    case 'critical': return '#dc2626'; // red-600
    case 'high': return '#ea580c'; // orange-600
    case 'moderate': return '#d97706'; // amber-600
    default: return '#059669'; // emerald-600
  }
}

function getRiskBgColor(severity: RiskSeverity): string {
  switch (severity) {
    case 'critical': return '#fef2f2'; // red-50
    case 'high': return '#fff7ed'; // orange-50
    case 'moderate': return '#fffbeb'; // amber-50
    default: return '#f0fdf4'; // emerald-50
  }
}

function createReportHTML(data: ReportData): string {
  const riskSeverity = getRiskSeverity(data.classInfo.riskLevel);
  const riskColor = getRiskColor(riskSeverity);
  const riskBgColor = getRiskBgColor(riskSeverity);
  const dateStr = data.time ? `${data.date} at ${data.time}` : data.date;
  const timestamp = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Sort scores descending
  const sortedScores = data.allScores ? [...data.allScores].sort((a, b) => b.score - a.score) : [];

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Times New Roman', Times, Georgia, serif;
      background: #ffffff;
      color: #1a1a1a;
      line-height: 1.6;
      padding: 40px 50px;
      width: 210mm;
      min-height: 297mm;
      position: relative;
    }

    body::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #0d9488 0%, #14b8a6 50%, #0d9488 100%);
      z-index: 1000;
    }

    body::after {
      content: 'CONFIDENTIAL MEDICAL REPORT';
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(-45deg);
      font-size: 72px;
      font-weight: 700;
      color: rgba(13, 148, 136, 0.03);
      letter-spacing: 8px;
      white-space: nowrap;
      z-index: -1;
      pointer-events: none;
    }

    .header {
      border-bottom: 3px solid #0d9488;
      padding-bottom: 20px;
      margin-bottom: 30px;
      position: relative;
    }

    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
    }

    .header-logo {
      font-size: 28px;
      font-weight: 700;
      color: #0d9488;
      letter-spacing: 2px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .header-date {
      text-align: right;
      font-size: 11px;
      color: #64748b;
      line-height: 1.4;
    }

    .header-title {
      font-size: 16px;
      font-weight: 700;
      color: #1a1a1a;
      text-transform: uppercase;
      letter-spacing: 2px;
      margin-bottom: 4px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .header-subtitle {
      font-size: 11px;
      color: #64748b;
      font-weight: 400;
      font-style: italic;
    }

    .info-bar {
      background: #f8fafc;
      border: 2px solid #e2e8f0;
      border-left: 5px solid #0d9488;
      padding: 20px 24px;
      margin-bottom: 28px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
    }

    .info-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-right: 20px;
      border-right: 1px solid #e2e8f0;
    }

    .info-item:last-child {
      border-right: none;
    }

    .info-label {
      font-size: 9px;
      font-weight: 700;
      color: #0d9488;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .info-value {
      font-size: 13px;
      font-weight: 600;
      color: #1a1a1a;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .section {
      margin-bottom: 28px;
      page-break-inside: avoid;
    }

    .section-title {
      font-size: 11px;
      font-weight: 700;
      color: #1a1a1a;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: 14px;
      padding-bottom: 6px;
      border-bottom: 2px solid #0d9488;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .diagnosis-card {
      background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
      border: 3px solid #0d9488;
      border-radius: 4px;
      padding: 28px 32px;
      margin-bottom: 28px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(13, 148, 136, 0.08);
    }

    .diagnosis-card::before {
      content: '';
      position: absolute;
      top: 0;
      right: 0;
      width: 150px;
      height: 150px;
      background: radial-gradient(circle, rgba(13, 148, 136, 0.05) 0%, transparent 70%);
      border-radius: 50%;
      transform: translate(40%, -40%);
    }

    .diagnosis-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 16px;
    }

    .diagnosis-main {
      flex: 1;
      position: relative;
      z-index: 1;
    }

    .diagnosis-name {
      font-size: 22px;
      font-weight: 700;
      color: #1a1a1a;
      margin-bottom: 10px;
      line-height: 1.3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .diagnosis-badge {
      display: inline-block;
      background: #ffffff;
      border: 2px solid #0d9488;
      padding: 5px 14px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 700;
      color: #0d9488;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .diagnosis-stats {
      text-align: right;
      min-width: 140px;
      position: relative;
      z-index: 1;
    }

    .confidence-label {
      font-size: 10px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-bottom: 6px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .confidence-value {
      font-size: 36px;
      font-weight: 700;
      color: #0d9488;
      line-height: 1;
      margin-bottom: 12px;
      letter-spacing: 3px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .risk-badge {
      display: inline-block;
      background: ${riskColor};
      color: white;
      padding: 6px 16px;
      border-radius: 3px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .two-column {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }

    .card {
      background: white;
      border: 2px solid #e2e8f0;
      border-radius: 4px;
      padding: 24px;
      height: 100%;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .image-box {
      background: #f8fafc;
      border: 2px solid #e2e8f0;
      border-radius: 4px;
      padding: 16px;
      text-align: center;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .lesion-image {
      max-width: 100%;
      max-height: 190px;
      border-radius: 2px;
      object-fit: contain;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .image-placeholder {
      color: #cbd5e1;
      font-size: 13px;
    }

    .detail-group {
      margin-bottom: 18px;
      padding-bottom: 16px;
      border-bottom: 1px solid #f1f5f9;
    }

    .detail-group:last-child {
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }

    .detail-label {
      font-size: 9px;
      font-weight: 700;
      color: #0d9488;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-bottom: 6px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .detail-value {
      font-size: 12px;
      color: #1a1a1a;
      line-height: 1.7;
    }

    .probability-card {
      background: white;
      border: 2px solid #e2e8f0;
      border-radius: 4px;
      padding: 24px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .probability-row {
      display: flex;
      align-items: center;
      margin-bottom: 12px;
      gap: 14px;
    }

    .probability-row:last-child {
      margin-bottom: 0;
    }

    .probability-label {
      font-size: 11px;
      font-weight: 500;
      color: #64748b;
      min-width: 150px;
      flex-shrink: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .probability-label.predicted {
      font-weight: 700;
      color: #1a1a1a;
    }

    .probability-bar-container {
      flex: 1;
      background: #f1f5f9;
      border-radius: 2px;
      height: 24px;
      position: relative;
      overflow: hidden;
      border: 1px solid #e2e8f0;
    }

    .probability-bar {
      background: linear-gradient(90deg, #cbd5e1 0%, #94a3b8 100%);
      height: 100%;
      border-radius: 1px;
      transition: width 0.3s ease;
    }

    .probability-bar.predicted {
      background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
      box-shadow: 0 2px 6px rgba(13, 148, 136, 0.2);
    }

    .probability-value {
      font-size: 11px;
      font-weight: 600;
      color: #64748b;
      min-width: 60px;
      text-align: right;
      flex-shrink: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      letter-spacing: 0.5px;
    }

    .probability-value.predicted {
      font-weight: 700;
      color: #0d9488;
    }

    .notes-card {
      background: #fffef7;
      border: 2px solid #e2e8f0;
      border-left: 5px solid #fbbf24;
      border-radius: 4px;
      padding: 24px;
      min-height: 120px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .notes-text {
      font-size: 12px;
      color: #1a1a1a;
      line-height: 1.8;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .notes-empty {
      font-size: 12px;
      color: #cbd5e1;
      font-style: italic;
    }

    .footer {
      margin-top: 50px;
      padding-top: 24px;
      border-top: 3px solid #0d9488;
      position: relative;
    }

    .disclaimer-title {
      font-size: 10px;
      font-weight: 700;
      color: #0d9488;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      margin-bottom: 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .disclaimer-text {
      font-size: 10px;
      color: #64748b;
      line-height: 1.7;
      margin-bottom: 16px;
      text-align: justify;
    }

    .footer-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid #e2e8f0;
      font-size: 9px;
      color: #94a3b8;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }

    .footer-logo {
      font-weight: 700;
      color: #0d9488;
      letter-spacing: 1.5px;
    }

    @media print {
      body {
        padding: 20px;
      }
    }
  </style>
</head>
<body>
  <!-- Header -->
  <div class="header">
    <div class="header-top">
      <div class="header-logo">DERMALYZE</div>
      <div class="header-date">
        <div><strong>Report Date:</strong> ${timestamp}</div>
        <div><strong>Document ID:</strong> ${data.caseId}</div>
      </div>
    </div>
    <div class="header-title">Dermatology Analysis Report</div>
    <div class="header-subtitle">AI-Assisted Skin Lesion Classification &amp; Clinical Assessment</div>
  </div>

  <!-- Case Information Bar -->
  <div class="info-bar">
    <div class="info-item">
      <div class="info-label">Case ID</div>
      <div class="info-value">${data.caseId}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Analysis Date</div>
      <div class="info-value">${dateStr}</div>
    </div>
    <div class="info-item">
      <div class="info-label">Clinician</div>
      <div class="info-value">${data.clinicianName}</div>
    </div>
  </div>

  <!-- Primary Diagnosis -->
  <div class="section">
    <div class="section-title">I. Primary Diagnosis</div>
    <div class="diagnosis-card">
      <div class="diagnosis-header">
        <div class="diagnosis-main">
          <div class="diagnosis-name">${data.className}</div>
          <div class="diagnosis-badge">${data.classId.toUpperCase()}</div>
        </div>
        <div class="diagnosis-stats">
          <div class="confidence-label">Confidence Score</div>
          <div class="confidence-value">${data.confidence.toFixed(1)} %</div>
          <div class="risk-badge">${data.classInfo.riskLevel.toUpperCase()}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Image and Clinical Details -->
  <div class="two-column">
    <!-- Lesion Image -->
    <div>
      <div class="section-title">II. Lesion Documentation</div>
      <div class="card">
        <div class="image-box">
          ${
            data.imageDataUrl
              ? `<img src="${data.imageDataUrl}" alt="Lesion" class="lesion-image" />`
              : '<div class="image-placeholder">No image available</div>'
          }
        </div>
      </div>
    </div>

    <!-- Clinical Details -->
    <div>
      <div class="section-title">III. Clinical Profile</div>
      <div class="card">
        <div class="detail-group">
          <div class="detail-label">Description</div>
          <div class="detail-value">${data.classInfo.description}</div>
        </div>
        <div class="detail-group">
          <div class="detail-label">Key Clinical Features</div>
          <div class="detail-value">${data.classInfo.keyFeatures}</div>
        </div>
        <div class="detail-group">
          <div class="detail-label">Common Demographics</div>
          <div class="detail-value">${data.classInfo.commonIn}</div>
        </div>
      </div>
    </div>
  </div>

  ${
    sortedScores.length > 0
      ? `
  <!-- Classification Probabilities -->
  <div class="section">
    <div class="section-title">IV. Differential Diagnosis Probabilities</div>
    <div class="probability-card">
      ${sortedScores
        .map((score) => {
          const isPredicted = score.id === data.classId;
          const barWidth = score.score;
          return `
        <div class="probability-row">
          <div class="probability-label ${isPredicted ? 'predicted' : ''}">${score.name}</div>
          <div class="probability-bar-container">
            <div class="probability-bar ${isPredicted ? 'predicted' : ''}" style="width: ${barWidth}%"></div>
          </div>
          <div class="probability-value ${isPredicted ? 'predicted' : ''}">${score.score.toFixed(1)} %</div>
        </div>
      `;
        })
        .join('')}
    </div>
  </div>
  `
      : ''
  }

  <!-- Clinician Notes -->
  <div class="section">
    <div class="section-title">V. Clinical Notes &amp; Observations</div>
    <div class="notes-card">
      ${
        data.notes && data.notes.trim()
          ? `<div class="notes-text">${data.notes}</div>`
          : '<div class="notes-empty">No clinical notes recorded for this analysis.</div>'
      }
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    <div class="disclaimer-title">⚠ Important Medical Disclaimer</div>
    <div class="disclaimer-text">
      This report is generated by an AI-assisted diagnostic tool for educational and screening purposes only.
      Classification results are probabilistic outputs based on machine learning algorithms and do not constitute
      a definitive medical diagnosis. All findings must be correlated with comprehensive clinical examination,
      patient history, dermoscopic evaluation, and histopathological analysis when clinically indicated.
      This report should only be interpreted by qualified healthcare professionals. For definitive diagnosis
      and treatment planning, consultation with a board-certified dermatologist is strongly recommended.
    </div>
    <div class="footer-meta">
      <div>Report Generated: ${timestamp}</div>
      <div class="footer-logo">POWERED BY DERMALYZE AI</div>
    </div>
  </div>
</body>
</html>
  `.trim();
}

export async function generateDermatologyReport(data: ReportData): Promise<void> {
  // Create a temporary container for the HTML
  const container = document.createElement('div');
  container.style.position = 'fixed';
  container.style.left = '-9999px';
  container.style.top = '0';
  container.style.width = '210mm';
  container.innerHTML = createReportHTML(data);
  document.body.appendChild(container);

  try {
    // Convert HTML to canvas
    const canvas = await html2canvas(container, {
      scale: 2, // Higher quality
      useCORS: true,
      allowTaint: true,
      backgroundColor: '#ffffff',
      logging: false,
      windowWidth: 794, // A4 width in pixels at 96 DPI (210mm)
      windowHeight: 1123, // A4 height in pixels at 96 DPI (297mm)
    });

    // Create PDF from canvas
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
    const imgX = (pdfWidth - imgWidth * ratio) / 2;

    pdf.addImage(imgData, 'PNG', imgX, 0, imgWidth * ratio, imgHeight * ratio);

    // Save the PDF
    const filename = `Dermalyze_Report_${data.caseId.replace(/[^a-zA-Z0-9]/g, '_')}.pdf`;
    pdf.save(filename);
  } finally {
    // Clean up the temporary container
    document.body.removeChild(container);
  }
}
