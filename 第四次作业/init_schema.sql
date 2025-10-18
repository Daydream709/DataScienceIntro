-- 创建数据库(如果尚未创建)
CREATE DATABASE IF NOT EXISTS esi_data CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE esi_data;

-- 创建学科表
CREATE TABLE IF NOT EXISTS subjects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建国家表
CREATE TABLE IF NOT EXISTS countries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    region VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建大学表
CREATE TABLE IF NOT EXISTS universities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    country_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (country_id) REFERENCES countries(id),
    UNIQUE KEY unique_university_country (name, country_id)
);

-- 创建排名表
CREATE TABLE IF NOT EXISTS rankings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    university_id INT NOT NULL,
    subject_id INT NOT NULL,
    year INT NOT NULL,
    rank_order INT NOT NULL,
    papers INT DEFAULT 0,
    citations INT DEFAULT 0,
    cites_per_paper DECIMAL(10, 3) DEFAULT 0.000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (university_id) REFERENCES universities(id),
    FOREIGN KEY (subject_id) REFERENCES subjects(id),
    UNIQUE KEY unique_ranking (university_id, subject_id, year)
);