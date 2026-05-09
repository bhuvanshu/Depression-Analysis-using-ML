package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.AdminSignupRequest;
import com.bhuvanshu.mindcare.service.AdminService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/admin")
@CrossOrigin("*")
public class AdminController {

    @Autowired
    private AdminService adminService;

    @PostMapping("/signup")
    public ResponseEntity<String> signup(
            @RequestBody AdminSignupRequest request) {

        String response = adminService.signup(request);

        return ResponseEntity.ok(response);
    }
}