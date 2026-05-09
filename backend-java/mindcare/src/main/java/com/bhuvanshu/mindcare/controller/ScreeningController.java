package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.ScreeningRequest;
import com.bhuvanshu.mindcare.dto.ScreeningResultResponse;
import com.bhuvanshu.mindcare.service.ScreeningService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/screening")
@CrossOrigin("*")
public class ScreeningController {

    @Autowired
    private ScreeningService screeningService;

    @PostMapping("/submit")
    public ResponseEntity<?> submitScreening(
            @RequestBody ScreeningRequest request) {

        ScreeningResultResponse response = screeningService.submitScreening(
                request);

        return ResponseEntity.ok(response);
    }
}