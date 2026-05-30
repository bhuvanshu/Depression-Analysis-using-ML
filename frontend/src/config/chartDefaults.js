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

export const getChartOptions = (theme, type = 'linear') => {
  const isLight = theme === 'light';
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: isLight ? '#475569' : '#94A3B8',
          font: { family: 'Inter', size: 12 },
          padding: 16
        }
      },
      tooltip: {
        backgroundColor: isLight ? '#FFFFFF' : '#1E293B',
        titleColor: isLight ? '#0F172A' : '#F1F5F9',
        bodyColor: isLight ? '#475569' : '#94A3B8',
        borderColor: isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.06)',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12
      }
    }
  };

  if (type === 'linear') {
    options.scales = {
      x: {
        grid: {
          color: isLight ? 'rgba(15, 23, 42, 0.05)' : 'rgba(255, 255, 255, 0.04)',
        },
        ticks: {
          color: isLight ? '#475569' : '#94A3B8',
          font: { family: 'Inter' }
        }
      },
      y: {
        grid: {
          color: isLight ? 'rgba(15, 23, 42, 0.05)' : 'rgba(255, 255, 255, 0.04)',
        },
        ticks: {
          color: isLight ? '#475569' : '#94A3B8',
          font: { family: 'Inter' }
        }
      }
    };
  }

  return options;
};
