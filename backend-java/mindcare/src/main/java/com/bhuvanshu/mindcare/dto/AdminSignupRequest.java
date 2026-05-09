package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AdminSignupRequest {

    private String collegeName;

    private String adminName;

    private String adminEmail;

    private String password;
}