import { 
  Chart as ChartJS, 
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, 
  Title, Tooltip, Legend, RadialLinearScale 
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';
import { Download, Printer, Filter } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import './ReportsPage.css';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, BarElement,
  RadialLinearScale, Title, Tooltip, Legend
);

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#94A3B8', font: { family: 'Inter' } } }
  }
};

export default function ReportsPage() {
  // Mock Data for Reports
  const stressVsRiskData = {
    labels: ['High Financial Stress', 'Medium Stress', 'Low Stress'],
    datasets: [
      {
        label: 'High Risk %',
        data: [65, 25, 10],
        backgroundColor: 'rgba(239, 68, 68, 0.8)',
        borderRadius: 4
      },
      {
        label: 'Moderate %',
        data: [25, 55, 30],
        backgroundColor: 'rgba(245, 158, 11, 0.8)',
        borderRadius: 4
      }
    ]
  };

  const radarData = {
    labels: ['Academic Pressure', 'Financial Stress', 'Sleep Quality', 'Study Satisfaction', 'Suicidal Thoughts'],
    datasets: [
      {
        label: 'High Risk Profile',
        data: [4.5, 4.2, 2.1, 1.8, 3.5], // Normalized 1-5
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 2,
      },
      {
        label: 'Low Risk Profile',
        data: [2.5, 2.0, 4.2, 4.0, 1.1],
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 2,
      }
    ]
  };

  return (
    <div className="reports-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Analytics Reports</h1>
          <p className="page-subtitle">Deep dive into institutional mental health metrics</p>
        </div>
        <div className="page-actions">
          <Button variant="secondary" icon={Printer}>Print</Button>
          <Button variant="primary" icon={Download}>Export PDF</Button>
        </div>
      </div>

      <div className="reports-toolbar">
        <Button variant="ghost" size="sm" icon={Filter}>Filter by Date</Button>
        <div className="report-date-range">Last 30 Days: Oct 1 - Oct 31, 2026</div>
      </div>

      <div className="reports-grid">
        <Card elevated className="report-card">
          <h3 className="report-card-title">Risk by Financial Stress</h3>
          <p className="report-card-desc">Correlation between financial stress levels and depression risk.</p>
          <div className="chart-container">
            <Bar 
              data={stressVsRiskData} 
              options={{
                ...chartDefaults,
                scales: {
                  x: { grid: { display: false }, ticks: { color: '#64748B' } },
                  y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748B' } }
                }
              }} 
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <h3 className="report-card-title">Profile Analysis</h3>
          <p className="report-card-desc">Average factor scores for High vs Low risk students.</p>
          <div className="chart-container">
            <Radar 
              data={radarData} 
              options={{
                ...chartDefaults,
                scales: {
                  r: {
                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    pointLabels: { color: '#94A3B8', font: { size: 11 } },
                    ticks: { display: false, min: 1, max: 5 }
                  }
                }
              }} 
            />
          </div>
        </Card>
      </div>
      
      <Card elevated className="report-insights">
        <h3 className="report-card-title">Key Insights</h3>
        <ul className="insights-list">
          <li><strong>Financial Stress correlation:</strong> Students with high financial stress are <strong>2.6x</strong> more likely to be classified as High Risk.</li>
          <li><strong>Academic Pressure:</strong> Across all risk categories, academic pressure remains the highest reported stressor (avg 4.1/5).</li>
          <li><strong>Departmental Anomaly:</strong> The Computer Science department shows a 15% month-over-month increase in moderate risk screenings.</li>
        </ul>
      </Card>
    </div>
  );
}
