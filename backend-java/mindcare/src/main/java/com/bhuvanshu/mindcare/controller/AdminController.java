package com.bhuvanshu.mindcare.controller;

import com.bhuvanshu.mindcare.dto.AdminLoginRequest;
import com.bhuvanshu.mindcare.dto.AdminLoginResponse;
import com.bhuvanshu.mindcare.dto.AdminSignupRequest;
import com.bhuvanshu.mindcare.service.AdminService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/admin")
public class AdminController {

    @Autowired
    private AdminService adminService;

    @PostMapping("/signup")
    public ResponseEntity<?> signup(
            @Valid @RequestBody AdminSignupRequest request) {
        try {
            String response = adminService.signup(request);
            return ResponseEntity.ok(response);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(
            @RequestBody AdminLoginRequest request) {

        AdminLoginResponse response = adminService.login(request);

        if (response != null) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.status(401).body("Invalid email or password");
        }
    }
}