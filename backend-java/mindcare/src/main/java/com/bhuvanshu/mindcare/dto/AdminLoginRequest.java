package com.bhuvanshu.mindcare.dto;
 
 import lombok.Getter;
 import lombok.Setter;
 
 @Getter
 @Setter
 public class AdminLoginRequest {
     private String adminEmail;
     private String password;
 }