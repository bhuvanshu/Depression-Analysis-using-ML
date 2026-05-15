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
@CrossOrigin("*")
public class AdminController {

    @Autowired
    private AdminService adminService;

    @PostMapping("/signup")
    public ResponseEntity<String> signup(
            @Valid @RequestBody AdminSignupRequest request) {

        String response = adminService.signup(request);

        return ResponseEntity.ok(response);
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