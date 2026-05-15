package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.service.DashboardService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/dashboard")
@CrossOrigin("*")
public class DashboardController {

    @Autowired
    private DashboardService dashboardService;

    // SUMMARY

    @GetMapping("/summary")
    public ResponseEntity<?> getSummary(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        return ResponseEntity.ok(
                dashboardService.getSummary(collegeName));
    }

    // STUDENT TABLE

    @GetMapping("/students")
    public ResponseEntity<?> getStudents(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        return ResponseEntity.ok(
                dashboardService.getAllStudents(collegeName));
    }

    // CHART DATA

    @GetMapping("/charts")
    public ResponseEntity<?> getCharts(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        return ResponseEntity.ok(
                dashboardService
                        .getRiskDistributionChart(collegeName));
    }

    // HIGH RISK STUDENTS

    @GetMapping("/high-risk")
    public ResponseEntity<?> getHighRiskStudents(
            @RequestHeader(value = "X-College-Name", required = false) String collegeName) {

        return ResponseEntity.ok(
                dashboardService
                        .getHighRiskStudents(collegeName));
    }
}