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
    public ResponseEntity<?> getSummary() {

        return ResponseEntity.ok(
                dashboardService.getSummary());
    }

    // STUDENT TABLE

    @GetMapping("/students")
    public ResponseEntity<?> getStudents() {

        return ResponseEntity.ok(
                dashboardService.getAllStudents());
    }
}