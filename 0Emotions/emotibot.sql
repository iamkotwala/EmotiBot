-- phpMyAdmin SQL Dump
-- version 4.8.4
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: May 06, 2020 at 12:00 PM
-- Server version: 5.7.24
-- PHP Version: 7.2.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `emotibot`
--

-- --------------------------------------------------------

--
-- Table structure for table `logs`
--

DROP TABLE IF EXISTS `logs`;
CREATE TABLE IF NOT EXISTS `logs` (
  `logID` int(5) NOT NULL AUTO_INCREMENT,
  `Emotion` varchar(10) NOT NULL,
  `Accuracy` double NOT NULL,
  `TimeStamp` varchar(35) NOT NULL,
  PRIMARY KEY (`logID`)
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `logs`
--

INSERT INTO `logs` (`logID`, `Emotion`, `Accuracy`, `TimeStamp`) VALUES
(2, 'Happy', 90, 'Wed May  6 17:26:30 2020'),
(3, 'Happy', 90, 'Wed May  6 17:26:30 2020'),
(4, 'Happy', 90, 'Wed May  6 17:26:30 2020'),
(5, 'Happy', 90, 'Wed May  6 17:26:30 2020'),
(6, 'Happy', 90, 'Wed May  6 17:26:30 2020'),
(7, 'Happy', 90, 'Wed May  6 17:26:31 2020'),
(8, 'Happy', 90, 'Wed May  6 17:26:31 2020'),
(9, 'Happy', 90, 'Wed May  6 17:26:31 2020'),
(10, 'Fearful', 89.9, 'Wed May  6 17:29:13 2020'),
(11, 'Surprised', 89.9, 'Wed May  6 17:29:13 2020'),
(12, 'Happy', 89.9, 'Wed May  6 17:29:13 2020'),
(13, 'Happy', 89.9, 'Wed May  6 17:29:13 2020'),
(14, 'Happy', 89.9, 'Wed May  6 17:29:13 2020'),
(15, 'Neutral', 89.9, 'Wed May  6 17:29:14 2020'),
(16, 'Neutral', 89.9, 'Wed May  6 17:29:14 2020'),
(17, 'Happy', 89.9, 'Wed May  6 17:29:14 2020');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
