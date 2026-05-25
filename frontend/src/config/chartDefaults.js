export const chartDefaults = {
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
