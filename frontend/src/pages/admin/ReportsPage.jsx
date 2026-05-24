import { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, BarElement, Title, Tooltip, Legend, Filler, ArcElement
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Download, FileText, TrendingUp, Brain, Info, Database, BarChart3, Activity } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import { getDashboardStudents } from '../../services/api';
import './ReportsPage.css';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler, ArcElement
);

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#94A3B8', font: { family: 'Inter', size: 12 }, padding: 16 }
    },
    tooltip: {
      backgroundColor: '#1E293B',
      titleColor: '#F1F5F9',
      bodyColor: '#94A3B8',
      borderColor: 'rgba(255,255,255,0.06)',
      borderWidth: 1,
      cornerRadius: 8,
      padding: 12
    }
  }
};

export default function ReportsPage() {
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboardStudents().then(data => {
      setStudents(data || []);
      setLoading(false);
    });
  }, []);

  // ── A. Average Stress Indicators ──
  const stressMetrics = useMemo(() => {
    if (!students.length) return { labels: [], datasets: [] };
    const sums = { academic: 0, financial: 0, satisfaction: 0 };
    students.forEach(s => {
      sums.academic += (s.academicPressure || 0);
      sums.financial += (s.financialStress || 0);
      sums.satisfaction += (s.studySatisfaction || 0);
    });
    const count = students.length;
    return {
      labels: ['Academic Pressure', 'Financial Stress', 'Study Satisfaction'],
      datasets: [{
        label: 'Institutional Avg (1-5)',
        data: [(sums.academic / count).toFixed(2), (sums.financial / count).toFixed(2), (sums.satisfaction / count).toFixed(2)],
        backgroundColor: ['rgba(99, 102, 241, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)'],
        borderRadius: 8,
      }]
    };
  }, [students]);
  // ── C. Department-wise Risk Bar ──
  const deptChartData = useMemo(() => {
    const deptMap = {};
    students.forEach(s => {
      const d = s.department || 'Unknown';
      if (!deptMap[d]) deptMap[d] = { high: 0, moderate: 0, low: 0 };
      
      if (s.riskLevel === 'High') deptMap[d].high++;
      else if (s.riskLevel === 'Moderate') deptMap[d].moderate++;
      else deptMap[d].low++;
    });

    const labels = Object.keys(deptMap);
    return {
      labels: labels.map(d => d.length > 12 ? d.slice(0, 12) + '…' : d),
      datasets: [
        {
          label: 'High Risk',
          data: labels.map(d => deptMap[d].high),
          backgroundColor: 'rgba(239, 68, 68, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Moderate',
          data: labels.map(d => deptMap[d].moderate),
          backgroundColor: 'rgba(245, 158, 11, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Low Risk',
          data: labels.map(d => deptMap[d].low),
          backgroundColor: 'rgba(34, 197, 94, 0.7)',
          borderRadius: 4
        }
      ]
    };
  }, [students]);

  // ── D. Study Satisfaction Distribution (Pie Chart) ──
  const satisfactionPieData = useMemo(() => {
    const counts = Array(6).fill(0);
    students.forEach(s => {
      const val = s.studySatisfaction;
      if (val !== undefined && val !== null && val >= 0 && val <= 5) {
        counts[val]++;
      }
    });

    return {
      labels: ['Very Low (0)', 'Low (1)', 'Below Avg (2)', 'Average (3)', 'Good (4)', 'Excellent (5)'],
      datasets: [{
        data: counts,
        backgroundColor: [
          'rgba(239, 68, 68, 0.75)',   // Very Low
          'rgba(249, 115, 22, 0.75)',  // Low
          'rgba(245, 158, 11, 0.75)',  // Below Avg
          'rgba(99, 102, 241, 0.75)',  // Average
          'rgba(59, 130, 246, 0.75)',  // Good
          'rgba(34, 197, 94, 0.75)'    // Excellent
        ],
        borderColor: [
          '#EF4444', '#F97316', '#F59E0B', '#6366F1', '#3B82F6', '#22C55E'
        ],
        borderWidth: 2
      }]
    };
  }, [students]);

  // ── E. Risk Distribution (Bar Chart) ──
  const riskBarData = useMemo(() => {
    let low = 0, moderate = 0, high = 0;
    students.forEach(s => {
      if (s.riskLevel === 'High') high++;
      else if (s.riskLevel === 'Moderate') moderate++;
      else if (s.riskLevel === 'Low') low++;
    });

    return {
      labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
      datasets: [{
        label: 'Student Count',
        data: [low, moderate, high],
        backgroundColor: [
          'rgba(34, 197, 94, 0.7)',
          'rgba(245, 158, 11, 0.7)',
          'rgba(239, 68, 68, 0.7)'
        ],
        borderColor: ['#22C55E', '#F59E0B', '#EF4444'],
        borderWidth: 2,
        borderRadius: 6
      }]
    };
  }, [students]);

  // ── F. Suicidal Thoughts Flag Rate (Pie Chart) ──
  const suicidalThoughtsPieData = useMemo(() => {
    let yesCount = 0;
    let noCount = 0;

    students.forEach(s => {
      if (s.suicidalThoughts === true || s.suicidalThoughts === 'Yes' || s.suicidalThoughts === 1) {
        yesCount++;
      } else {
        noCount++;
      }
    });

    return {
      labels: ['Reported Thoughts (Yes)', 'No Reported Thoughts (No)'],
      datasets: [{
        data: [yesCount, noCount],
        backgroundColor: [
          'rgba(239, 68, 68, 0.75)',  // Red
          'rgba(34, 197, 94, 0.75)'   // Green
        ],
        borderColor: [
          '#EF4444',
          '#22C55E'
        ],
        borderWidth: 2
      }]
    };
  }, [students]);

  return (
    <div className="reports-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Intelligence Reports</h1>
          <p className="page-subtitle">Analytical & Export Layer — Institutional Data Patterns</p>
        </div>
        <div className="page-actions">
          <Button variant="secondary" icon={FileText}>Export CSV</Button>
          <Button variant="primary" icon={Download}>Export PDF</Button>
        </div>
      </div>

      <div className="reports-grid">
        <Card elevated className="report-card">
          <div className="report-card-header">
            <TrendingUp size={18} style={{ color: 'var(--accent-primary)' }} />
            <h3>Average Stress Indicators</h3>
          </div>
          <div className="report-chart-container">
            <Bar
              data={stressMetrics}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <Activity size={18} style={{ color: 'var(--accent-warning)' }} />
            <h3>Study Satisfaction Distribution</h3>
          </div>
          <div className="report-chart-container">
            <Pie
              data={satisfactionPieData}
              options={{
                ...chartDefaults,
                plugins: {
                  ...chartDefaults.plugins,
                  legend: {
                    position: 'bottom',
                    onClick: () => {},
                    labels: {
                      color: '#94A3B8',
                      font: { family: 'Inter', size: 11 }
                    }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <TrendingUp size={18} style={{ color: 'var(--accent-danger)' }} />
            <h3>Risk Distribution Breakdown</h3>
          </div>
          <div className="report-chart-container">
            <Bar
              data={riskBarData}
              options={{
                ...chartDefaults,
                scales: {
                  x: {
                    grid: { display: false },
                    ticks: { color: '#64748B', font: { size: 11 } }
                  },
                  y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748B', stepSize: 1 }
                  }
                },
                plugins: {
                  ...chartDefaults.plugins,
                  legend: { display: false }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card">
          <div className="report-card-header">
            <Brain size={18} style={{ color: 'var(--accent-danger)' }} />
            <h3>Suicidal Thoughts Flag Rate</h3>
          </div>
          <div className="report-chart-container">
            <Pie
              data={suicidalThoughtsPieData}
              options={{
                ...chartDefaults,
                plugins: {
                  ...chartDefaults.plugins,
                  legend: {
                    position: 'bottom',
                    onClick: () => {},
                    labels: {
                      color: '#94A3B8',
                      font: { family: 'Inter', size: 11 }
                    }
                  }
                }
              }}
            />
          </div>
        </Card>

        <Card elevated className="report-card report-card-full">
          <div className="report-card-header">
            <BarChart3 size={18} style={{ color: 'var(--accent-primary)' }} />
            <h3>Department-wise Risk Breakdown</h3>
          </div>
          <div className="report-chart-container">
            <Bar
              data={deptChartData}
              options={{
                ...chartDefaults,
                scales: {
                  x: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { color: '#64748B', font: { size: 11 } }
                  },
                  y: {
                    stacked: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748B', stepSize: 1 }
                  }
                }
              }}
            />
          </div>
        </Card>
      </div>

      <Card elevated className="intelligence-summary">
        <div className="summary-header">
          <Info size={20} />
          <h3>System Intelligence Summary</h3>
        </div>
        <div className="summary-grid">
          <div className="summary-item">
            <label>Primary Stressor</label>
            <span>Academic Pressure (Avg 4.2)</span>
          </div>
          <div className="summary-item">
            <label>Top Risk Department</label>
            <span style={{ color: 'var(--accent-danger)' }}>Computer Science</span>
          </div>
          <div className="summary-item">
            <label>Participation Rate</label>
            <span>82.4% institutional coverage</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
