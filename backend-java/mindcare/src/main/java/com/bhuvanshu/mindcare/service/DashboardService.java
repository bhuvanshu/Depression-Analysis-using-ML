package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.DashboardStudentResponse;
import com.bhuvanshu.mindcare.entity.ScreeningResult;
import com.bhuvanshu.mindcare.entity.Student;
import com.bhuvanshu.mindcare.repository.ScreeningResultRepository;
import com.bhuvanshu.mindcare.repository.StudentRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class DashboardService {

    @Autowired
    private StudentRepository studentRepository;

    @Autowired
    private ScreeningResultRepository screeningResultRepository;

    // SUMMARY API

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

    // STUDENT TABLE API

    public List<DashboardStudentResponse> getAllStudents() {

        List<ScreeningResult> results = screeningResultRepository
                .findAllByOrderByPredictedAtDesc();

        List<DashboardStudentResponse> responseList = new ArrayList<>();

        for (ScreeningResult result : results) {

            Student student = result.getScreeningResponse()
                    .getStudent();

            DashboardStudentResponse dto = new DashboardStudentResponse();

            dto.setEnrollmentId(
                    student.getEnrollmentId());

            dto.setStudentName(
                    student.getName());

            dto.setDepartment(
                    student.getDepartment());

            dto.setRiskLevel(
                    result.getRiskLevel());

            dto.setProbabilityScore(
                    result.getProbabilityScore());

            responseList.add(dto);
        }

        return responseList;
    }
}