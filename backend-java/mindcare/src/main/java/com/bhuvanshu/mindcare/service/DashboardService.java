package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.repository.ScreeningResultRepository;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class DashboardService {

    @Autowired
    private StudentRepository studentRepository;

    @Autowired
    private ScreeningResultRepository screeningResultRepository;

    public Map<String, Object> getSummary() {

        Map<String, Object> summary = new HashMap<>();

        summary.put(
                "totalStudents",
                studentRepository.count());

        summary.put(
                "highRisk",
                screeningResultRepository
                        .countByRiskLevel("High"));

        summary.put(
                "moderateRisk",
                screeningResultRepository
                        .countByRiskLevel("Moderate"));

        summary.put(
                "lowRisk",
                screeningResultRepository
                        .countByRiskLevel("Low"));

        return summary;
    }
}