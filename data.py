import numpy as np

gerontidou_2010 = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.03,0.05,0.05,0.06,0.07,0.07,0.07,0.07,0.07,0.06,0.05,0.04],
[0.11,0.07,0.03,0.00,0.00,0.00,0.00,0.00,0.01,0.04,0.07,0.09,0.12,0.14,0.16,0.17,0.18,0.18,0.19,0.20,0.21,0.20,0.18,0.15],
[0.29,0.20,0.12,0.06,0.03,0.01,0.02,0.04,0.08,0.13,0.19,0.25,0.31,0.34,0.37,0.39,0.40,0.42,0.44,0.47,0.48,0.49,0.46,0.39],
[0.66,0.46,0.30,0.17,0.10,0.08,0.09,0.13,0.22,0.34,0.47,0.58,0.66,0.72,0.75,0.76,0.78,0.82,0.87,0.92,0.99,1.01,0.96,0.83],
[1.26,0.94,0.63,0.40,0.26,0.21,0.23,0.33,0.51,0.73,0.97,1.13,1.27,1.32,1.36,1.38,1.43,1.47,1.56,1.66,1.78,1.85,1.77,1.57],
[2.16,1.65,1.18,0.79,0.55,0.45,0.48,0.68,1.00,1.40,1.78,2.02,2.15,2.22,2.25,2.32,2.36,2.45,2.59,2.77,2.96,3.04,2.95,2.64],
[3.29,2.67,2.01,1.43,1.01,0.85,0.92,1.26,1.82,2.50,2.96,3.27,3.40,3.46,3.49,3.58,3.69,3.84,4.03,4.26,4.56,4.65,4.49,4.00],
[4.74,4.02,3.09,2.32,1.73,1.45,1.53,2.05,2.95,4.00,4.62,4.90,5.00,5.03,5.10,5.22,5.41,5.56,5.78,6.03,6.29,6.32,5.92,5.34],
[6.32,5.38,4.54,3.51,2.68,2.25,2.42,3.17,4.53,5.63,6.61,7.09,7.14,7.04,7.06,7.32,7.69,8.05,8.39,8.88,9.23,9.17,8.56,7.55],
[8.82,7.54,5.92,4.96,3.92,3.29,3.52,4.60,6.31,8.66,9.53,9.74,9.63,9.74,10.00,10.50,10.89,11.20,11.05,11.28,11.44,11.09,10.25,9.35],
[10.35,9.58,8.41,6.56,5.20,4.28,4.58,6.26,9.20,10.85,11.34,11.59,11.66,11.74,12.09,12.61,13.10,13.61,13.87,13.86,13.65,13.17,12.48,11.51],
[12.13,11.40,10.43,9.03,7.17,5.89,6.25,8.70,11.22,12.35,12.98,13.34,13.59,13.82,14.15,14.61,15.07,15.36,15.39,15.20,14.81,14.21,13.51,12.80],
[13.06,12.46,11.71,10.52,8.26,7.06,7.55,10.62,12.35,13.26,13.85,14.28,14.61,14.89,15.25,15.76,16.26,16.53,16.48,16.17,15.66,15.00,14.31,13.65],
[13.79,13.28,12.66,11.68,10.17,9.11,10.00,11.92,12.98,13.72,14.25,14.70,15.11,15.48,15.93,16.50,17.02,17.28,17.19,16.80,16.23,15.57,14.91,14.32],
[14.35,13.90,13.38,12.66,11.67,10.97,11.57,12.50,13.28,13.82,14.25,14.71,15.18,15.64,16.18,16.82,17.37,17.64,17.53,17.11,16.53,15.91,15.34,14.82],
[14.72,14.30,13.85,13.31,12.59,12.10,12.24,12.77,13.30,13.61,13.90,14.33,14.83,15.38,16.03,16.74,17.32,17.60,17.50,17.09,16.55,16.01,15.55,15.13],
[14.86,14.49,14.09,13.64,13.11,12.63,12.52,12.81,13.06,13.12,13.24,13.60,14.11,14.73,15.49,16.26,16.87,17.16,17.10,16.74,16.27,15.85,15.52,15.20],
[14.76,14.46,14.12,13.73,13.27,12.80,12.58,12.64,12.63,12.42,12.36,12.60,13.08,13.77,14.61,15.42,16.02,16.32,16.32,16.05,15.68,15.39,15.20,15.01],
[14.39,14.20,13.94,13.62,13.22,12.76,12.44,12.30,12.01,11.53,11.22,11.38,11.85,12.57,13.45,14.25,14.80,15.09,15.15,14.97,14.71,14.55,14.53,14.52],
[13.73,13.69,13.55,13.32,12.99,12.55,12.13,11.80,11.28,10.53,9.85,9.96,10.35,11.13,12.02,12.67,13.19,13.42,13.53,13.45,13.29,13.30,13.45,13.65],
[12.70,12.91,12.95,12.85,12.59,12.18,11.67,11.17,10.31,9.25,8.43,8.22,8.65,9.29,9.99,10.50,10.79,10.93,11.11,11.04,10.52,10.63,11.63,12.32],
[10.62,11.56,12.12,12.19,12.04,11.68,11.09,10.31,9.29,7.96,7.01,6.85,7.13,7.65,8.11,8.18,7.93,7.65,7.70,7.81,7.95,8.48,9.33,9.70],
[8.95,8.87,10.49,11.32,11.35,11.06,10.32,9.43,8.15,6.73,5.99,5.50,5.56,5.82,5.84,5.61,5.37,5.25,5.19,5.27,5.45,5.82,6.48,7.62],
[6.25,7.64,8.06,9.69,10.49,10.19,9.49,8.41,6.94,5.88,5.03,4.40,4.24,4.30,4.30,4.11,3.80,3.44,3.39,3.43,3.65,4.14,4.78,5.37],
[4.54,5.32,6.50,7.95,9.35,9.21,8.53,7.35,6.26,5.16,4.08,3.58,3.36,3.35,3.19,2.78,2.46,2.15,2.06,2.10,2.23,2.61,3.17,3.95],
[3.14,4.10,4.78,5.88,7.54,8.15,7.67,6.73,5.48,4.16,3.36,2.91,2.69,2.47,2.24,1.89,1.53,1.26,1.11,1.13,1.24,1.52,1.97,2.56],
[2.16,2.79,3.65,4.43,5.24,6.19,6.20,5.31,4.27,3.40,2.75,2.34,2.08,1.83,1.56,1.22,0.91,0.67,0.55,0.54,0.61,0.80,1.11,1.57],
[1.32,1.87,2.50,3.22,4.02,4.46,4.47,4.08,3.45,2.85,2.25,1.86,1.58,1.33,1.06,0.78,0.51,0.33,0.24,0.22,0.26,0.37,0.57,0.89],
[0.75,1.13,1.61,2.19,2.76,3.30,3.45,3.21,2.66,2.18,1.76,1.44,1.19,0.93,0.69,0.46,0.28,0.15,0.09,0.07,0.08,0.14,0.26,0.46],
[0.40,0.67,1.01,1.42,1.82,2.17,2.32,2.21,1.95,1.60,1.32,1.06,0.85,0.64,0.44,0.27,0.14,0.06,0.01,0.00,0.00,0.04,0.10,0.22],
[0.21,0.38,0.59,0.85,1.11,1.34,1.45,1.43,1.31,1.12,0.92,0.74,0.57,0.41,0.28,0.15,0.07,0.02,0.00,0.00,0.00,0.00,0.03,0.10],
[0.11,0.20,0.33,0.48,0.63,0.77,0.85,0.86,0.81,0.72,0.60,0.49,0.36,0.26,0.17,0.09,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.05],
[0.06,0.12,0.18,0.26,0.34,0.40,0.44,0.46,0.44,0.40,0.35,0.29,0.22,0.16,0.10,0.06,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.03],
[0.06,0.08,0.11,0.14,0.16,0.19,0.21,0.21,0.21,0.20,0.18,0.15,0.13,0.10,0.08,0.06,0.04,0.03,0.02,0.01,0.01,0.02,0.03,0.04],
[0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08],
])

gerontidou_2015 = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],[0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.03,0.04,0.05,0.05,0.06,0.06,0.06,0.07,0.06,0.06,0.05,0.04],[0.10,0.07,0.03,0.00,0.00,0.00,0.00,0.00,0.01,0.04,0.07,0.09,0.12,0.14,0.15,0.16,0.16,0.17,0.18,0.19,0.19,0.19,0.17,0.14],[0.29,0.19,0.12,0.06,0.03,0.02,0.02,0.05,0.08,0.14,0.20,0.25,0.30,0.34,0.36,0.37,0.38,0.40,0.42,0.45,0.46,0.47,0.44,0.37],[0.64,0.46,0.29,0.18,0.11,0.09,0.10,0.14,0.23,0.35,0.48,0.58,0.65,0.70,0.72,0.75,0.76,0.79,0.83,0.89,0.94,0.97,0.92,0.81],[1.25,0.91,0.63,0.41,0.27,0.22,0.24,0.35,0.53,0.76,0.96,1.15,1.26,1.31,1.33,1.35,1.38,1.41,1.49,1.63,1.74,1.79,1.74,1.53],[2.13,1.63,1.17,0.81,0.57,0.47,0.52,0.71,1.06,1.47,1.82,2.04,2.13,2.19,2.22,2.27,2.29,2.36,2.51,2.70,2.88,3.00,2.88,2.59],[3.24,2.66,2.00,1.43,1.04,0.87,0.95,1.30,1.89,2.58,3.00,3.28,3.39,3.41,3.43,3.48,3.60,3.70,3.91,4.19,4.45,4.60,4.43,3.95],[4.70,3.99,3.09,2.32,1.75,1.48,1.60,2.17,3.07,4.11,4.67,4.94,4.99,4.98,5.02,5.14,5.31,5.46,5.65,5.94,6.21,6.25,5.86,5.32],[6.25,5.33,4.51,3.48,2.69,2.31,2.48,3.34,4.72,5.77,6.74,7.19,7.16,6.98,6.96,7.21,7.51,7.81,8.17,8.74,9.12,9.09,8.51,7.49],[8.74,7.44,5.86,4.89,3.87,3.36,3.62,4.76,6.61,8.88,9.64,9.78,9.66,9.70,9.93,10.40,10.73,10.89,10.95,11.16,11.38,11.04,10.21,9.34],[10.28,9.49,8.31,6.50,5.18,4.35,4.77,6.62,9.54,10.91,11.45,11.65,11.68,11.73,12.04,12.52,12.91,13.39,13.76,13.79,13.61,13.14,12.45,11.47],[12.06,11.30,10.32,8.91,7.09,5.93,6.48,9.11,11.39,12.46,13.06,13.39,13.62,13.82,14.14,14.57,15.01,15.28,15.31,15.15,14.79,14.20,13.49,12.76],[12.98,12.36,11.59,10.32,8.13,6.81,7.57,10.98,12.46,13.33,13.90,14.32,14.63,14.90,15.25,15.75,16.22,16.48,16.43,16.14,15.65,14.99,14.29,13.60],[13.72,13.18,12.54,11.53,10.06,9.22,10.27,12.03,13.05,13.76,14.28,14.73,15.13,15.49,15.94,16.51,17.01,17.26,17.17,16.79,16.23,15.56,14.89,14.28],[14.28,13.80,13.26,12.53,11.57,10.99,11.67,12.56,13.32,13.84,14.27,14.73,15.19,15.65,16.21,16.85,17.40,17.65,17.53,17.11,16.53,15.90,15.31,14.78],[14.65,14.21,13.74,13.19,12.51,12.07,12.25,12.79,13.29,13.59,13.89,14.33,14.84,15.40,16.07,16.79,17.37,17.64,17.52,17.10,16.55,16.00,15.52,15.08],[14.80,14.41,13.99,13.53,13.01,12.55,12.48,12.79,13.02,13.07,13.20,13.58,14.10,14.76,15.54,16.33,16.94,17.22,17.14,16.77,16.28,15.84,15.48,15.16],[14.71,14.39,14.03,13.63,13.16,12.70,12.51,12.58,12.54,12.32,12.29,12.56,13.07,13.80,14.67,15.51,16.12,16.41,16.39,16.09,15.68,15.37,15.16,14.97],[14.34,14.13,13.86,13.52,13.11,12.65,12.34,12.19,11.89,11.39,11.12,11.32,11.83,12.60,13.52,14.34,14.91,15.20,15.24,15.03,14.71,14.53,14.50,14.48],[13.69,13.63,13.48,13.22,12.87,12.43,12.01,11.67,11.12,10.37,9.73,9.88,10.32,11.16,12.07,12.78,13.31,13.55,13.66,13.52,13.31,13.28,13.41,13.62],[12.67,12.86,12.88,12.75,12.48,12.04,11.53,11.00,10.11,9.04,8.26,8.14,8.63,9.31,10.04,10.56,10.80,11.13,11.29,11.19,10.51,10.62,11.62,12.29],[10.63,11.56,12.06,12.10,11.93,11.54,10.91,10.12,9.06,7.75,6.85,6.73,7.12,7.67,8.14,8.27,8.01,7.80,7.89,7.87,7.91,8.48,9.28,9.66],[8.96,8.72,10.49,11.24,11.23,10.90,10.14,9.22,7.91,6.54,5.82,5.43,5.54,5.83,5.88,5.63,5.46,5.34,5.30,5.36,5.47,5.82,6.46,7.61],[6.26,7.65,8.04,9.69,10.38,10.03,9.30,8.17,6.72,5.67,4.87,4.32,4.21,4.29,4.30,4.12,3.85,3.52,3.48,3.50,3.70,4.16,4.79,5.39],[4.53,5.32,6.51,7.89,9.25,9.04,8.30,7.09,6.02,4.95,3.96,3.51,3.34,3.36,3.25,3.01,3.00,3.00,2.13,2.17,3.00,3.00,3.21,3.97],[3.22,4.08,4.76,5.87,7.47,8.01,7.49,6.56,5.22,4.00,3.28,3.05,2.68,2.46,2.24,2.03,2.00,2.00,1.16,1.16,2.00,2.00,2.06,3.00],[2.18,3.04,3.64,4.43,5.18,6.05,5.98,5.07,4.13,3.30,2.66,2.30,2.07,1.81,1.54,1.21,1.00,1.00,0.57,0.55,1.00,1.00,1.12,2.00],[1.31,2.02,3.00,3.24,3.96,4.38,4.36,3.96,3.35,2.76,2.19,1.84,1.56,1.31,1.05,0.76,0.51,0.34,0.24,0.22,0.26,0.37,0.58,1.00],[0.76,1.14,2.00,2.17,3.00,3.24,3.40,3.18,2.55,2.12,1.72,1.41,1.16,0.92,0.68,0.46,0.27,0.15,0.09,0.07,0.09,0.14,0.26,0.47],[0.41,0.67,1.02,1.41,2.00,2.12,2.27,2.16,2.01,1.54,1.27,1.02,0.82,0.63,0.43,0.27,0.14,0.06,0.01,0.00,0.00,0.04,0.11,0.22],[0.21,0.38,0.58,0.84,1.09,1.31,1.42,1.39,1.28,1.08,0.89,0.72,0.55,0.41,0.27,0.15,0.07,0.01,0.00,0.00,0.00,0.00,0.03,0.10],[0.11,0.20,0.33,0.47,0.62,0.75,0.82,0.83,0.79,0.69,0.58,0.47,0.36,0.25,0.16,0.09,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.05],[0.07,0.12,0.18,0.26,0.33,0.39,0.43,0.45,0.44,0.39,0.35,0.28,0.22,0.15,0.10,0.06,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.03],[0.06,0.08,0.10,0.13,0.16,0.18,0.20,0.21,0.20,0.19,0.18,0.15,0.13,0.10,0.08,0.06,0.04,0.02,0.02,0.01,0.01,0.02,0.02,0.04],[0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.07,0.08,0.08,0.08,0.08]])

gerontidou_2020 = np.array(
[
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],[0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.03,0.04,0.04,0.05,0.06,0.06,0.06,0.06,0.06,0.05,0.05,0.03],[0.10,0.06,0.03,0.00,0.00,0.00,0.00,0.00,0.02,0.04,0.07,0.09,0.11,0.13,0.14,0.15,0.16,0.16,0.17,0.18,0.18,0.18,0.16,0.13],[0.28,0.19,0.11,0.06,0.03,0.02,0.03,0.05,0.09,0.14,0.20,0.26,0.30,0.33,0.35,0.35,0.36,0.38,0.40,0.42,0.45,0.45,0.42,0.36],[0.62,0.45,0.29,0.18,0.12,0.09,0.11,0.16,0.25,0.36,0.49,0.59,0.65,0.69,0.71,0.72,0.73,0.76,0.79,0.86,0.91,0.93,0.90,0.79],[1.21,0.91,0.62,0.41,0.28,0.24,0.26,0.37,0.55,0.78,0.98,1.15,1.26,1.29,1.31,1.33,1.34,1.38,1.44,1.59,1.69,1.75,1.68,1.49],[2.10,1.62,1.16,0.81,0.58,0.50,0.54,0.76,1.09,1.53,1.87,2.07,2.14,2.17,2.18,2.20,2.24,2.29,2.44,2.65,2.84,2.95,2.84,2.54],[3.22,2.63,2.01,1.43,1.07,0.92,1.01,1.37,1.98,2.65,3.08,3.33,3.43,3.40,3.40,3.42,3.52,3.62,3.81,4.15,4.38,4.56,4.36,3.93],[4.66,3.97,3.08,2.33,1.77,1.52,1.70,2.29,3.21,4.22,4.76,4.97,5.02,4.99,4.97,5.10,5.20,5.35,5.56,5.85,6.13,6.21,5.83,5.30],[6.20,5.29,4.49,3.48,2.72,2.36,2.60,3.59,4.84,5.94,6.91,7.25,7.17,6.95,6.91,7.10,7.35,7.59,7.95,8.62,9.09,9.04,8.49,7.42],[8.65,7.33,5.81,4.84,3.83,3.41,3.72,4.97,6.94,9.08,9.75,9.80,9.67,9.67,9.88,10.30,10.58,10.83,10.95,11.06,11.34,10.99,10.17,9.32],[10.19,9.28,8.17,6.42,5.13,4.41,4.96,7.06,9.81,11.00,11.56,11.72,11.71,11.73,12.01,12.44,12.74,13.16,13.65,13.73,13.59,13.13,12.42,11.42],[11.97,11.18,10.19,8.73,7.00,5.99,6.77,9.31,11.55,12.56,13.14,13.46,13.65,13.83,14.13,14.55,14.95,15.20,15.24,15.11,14.77,14.19,13.47,12.70],[12.90,12.24,11.45,10.11,8.02,6.65,7.79,11.22,12.56,13.40,13.96,14.36,14.66,14.91,15.25,15.74,16.19,16.43,16.38,16.12,15.65,14.99,14.27,13.56],[13.64,13.07,12.41,11.39,9.95,9.31,10.50,12.13,13.12,13.81,14.33,14.77,15.15,15.51,15.96,16.52,17.01,17.24,17.14,16.78,16.23,15.56,14.87,14.23],[14.21,13.70,13.15,12.40,11.49,11.00,11.73,12.61,13.35,13.86,14.29,14.75,15.21,15.67,16.24,16.89,17.42,17.66,17.53,17.11,16.53,15.89,15.29,14.73],[14.58,14.12,13.64,13.08,12.42,12.04,12.25,12.80,13.29,13.58,13.89,14.33,14.84,15.42,16.11,16.84,17.42,17.67,17.54,17.12,16.55,15.99,15.50,15.04],[14.74,14.33,13.90,13.43,12.91,12.48,12.45,12.76,12.98,13.02,13.17,13.56,14.10,14.78,15.60,16.40,17.01,17.28,17.19,16.79,16.28,15.82,15.46,15.12],[14.66,14.32,13.94,13.53,13.06,12.61,12.44,12.52,12.47,12.24,12.21,12.52,13.06,13.83,14.74,15.59,16.20,16.50,16.45,16.12,15.68,15.34,15.14,14.93],[14.30,14.07,13.78,13.43,13.00,12.55,12.25,12.10,11.79,11.27,11.03,11.25,11.82,12.63,13.59,14.43,15.01,15.31,15.33,15.07,14.71,14.50,14.47,14.45],[13.65,13.58,13.41,13.14,12.77,12.31,11.90,11.55,10.97,10.14,9.59,9.80,10.31,11.19,12.11,12.87,13.42,13.69,13.78,13.58,13.30,13.25,13.38,13.60],[12.66,12.82,12.82,12.66,12.37,11.92,11.40,10.82,9.92,8.86,8.11,8.06,8.60,9.35,10.10,10.64,11.02,11.34,11.49,11.04,10.50,10.58,11.61,12.27],[10.67,11.58,12.00,12.01,11.82,11.41,10.74,9.94,8.83,7.58,6.72,6.66,7.11,7.70,8.17,8.31,8.12,8.02,8.11,8.02,7.94,8.46,9.26,9.63],[8.99,8.75,10.52,11.16,11.13,10.73,9.96,9.02,7.70,6.35,5.68,5.35,5.55,5.84,5.87,5.69,5.51,5.44,5.41,5.45,5.50,5.83,6.46,7.63],[6.27,7.69,7.95,9.68,10.28,9.87,9.13,7.96,6.52,5.56,4.73,4.26,4.20,4.28,4.30,4.14,3.90,3.62,3.57,3.59,3.73,4.17,4.82,5.39],[4.49,5.34,6.51,7.85,9.14,8.88,8.12,6.87,5.88,4.75,3.86,3.45,3.34,3.35,3.24,3.01,3.00,3.00,3.00,3.00,3.00,3.00,3.21,3.99],[3.25,4.09,4.76,5.86,7.43,7.88,7.34,6.41,5.01,3.88,3.20,2.78,2.63,2.44,2.24,2.04,2.00,2.00,2.00,2.00,2.00,2.00,2.07,3.00],[2.18,3.03,3.67,4.42,5.13,5.90,5.76,4.90,3.99,3.21,2.65,2.23,2.05,1.80,1.52,1.21,1.00,1.00,1.00,1.00,1.00,1.00,1.15,2.00],[1.32,2.03,3.00,3.25,3.92,4.31,4.27,3.86,3.24,2.65,2.14,1.81,1.53,1.29,1.05,0.75,0.51,0.33,0.25,0.23,0.27,0.38,0.59,1.00],[0.77,1.14,2.00,2.17,3.00,3.17,3.32,3.07,2.48,2.04,1.65,1.36,1.13,0.90,0.68,0.45,0.27,0.15,0.09,0.07,0.09,0.15,0.27,0.48],[0.42,0.68,1.02,1.40,2.00,2.09,2.23,2.13,1.80,1.50,1.22,1.00,0.80,0.61,0.42,0.26,0.13,0.06,0.01,0.00,0.00,0.04,0.11,0.23],[0.21,0.37,0.59,0.83,1.07,1.28,1.37,1.34,1.23,1.04,0.88,0.70,0.55,0.40,0.26,0.15,0.06,0.01,0.00,0.00,0.00,0.00,0.03,0.10],[0.11,0.20,0.33,0.47,0.62,0.73,0.81,0.81,0.75,0.67,0.57,0.45,0.35,0.25,0.16,0.08,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.05],[0.07,0.12,0.18,0.25,0.32,0.38,0.42,0.44,0.42,0.38,0.33,0.27,0.21,0.15,0.10,0.06,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.03],[0.06,0.08,0.10,0.13,0.16,0.18,0.19,0.20,0.20,0.19,0.17,0.15,0.12,0.10,0.07,0.05,0.04,0.02,0.01,0.00,0.00,0.01,0.02,0.04],[0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.08,0.08,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.08,0.07,0.07,0.07,0.07,0.07]
]
)

smartshea_2015 = np.array(
[[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.02,0.02,0.03,0.03,0.04,0.03,0.03,0.04,0.03,0.00
],[0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.05,0.04,0.07,0.09,0.08,0.13,0.08,0.16,0.14,0.13,0.15,0.17,0.13,0.08
],[0.21,0.11,0.04,0.00,0.00,0.00,0.00,0.02,0.08,0.15,0.20,0.18,0.28,0.34,0.32,0.35,0.41,0.38,0.42,0.47,0.47,0.46,0.43,0.33
],[0.55,0.30,0.19,0.07,0.04,0.05,0.06,0.15,0.31,0.45,0.58,0.60,0.66,0.74,0.75,0.76,0.79,0.81,0.87,0.91,0.98,1.01,0.96,0.72
],[1.08,0.74,0.48,0.32,0.21,0.22,0.31,0.49,0.70,0.94,1.14,1.13,1.30,1.35,1.39,1.39,1.44,1.47,1.56,1.72,1.76,1.86,1.80,1.42
],[1.85,1.39,0.93,0.65,0.50,0.50,0.65,0.94,1.38,1.75,2.02,2.10,2.18,2.22,2.30,2.30,2.38,2.45,2.59,2.75,2.96,3.10,2.97,2.31
],[2.91,2.27,1.60,1.18,0.93,0.90,1.21,1.71,2.40,2.96,3.24,3.40,3.49,3.49,3.53,3.58,3.69,3.81,4.05,4.30,4.61,4.72,4.54,3.59
],[4.31,3.42,2.68,1.96,1.57,1.56,1.94,2.85,3.92,4.60,4.97,4.98,5.09,5.07,5.12,5.25,5.39,5.53,5.77,6.03,6.30,6.35,5.96,4.96
],[5.65,4.88,4.06,2.99,2.46,2.43,3.00,4.24,5.47,6.56,7.24,7.33,7.26,7.11,7.08,7.32,7.67,7.99,8.34,8.93,9.30,9.22,8.63,6.72
],[8.11,6.43,5.30,4.31,3.54,3.50,4.29,5.87,8.43,9.54,9.86,9.87,9.73,9.83,10.05,10.51,10.87,11.13,11.02,11.29,11.51,11.16,10.30,9.33
],[9.78,8.85,7.24,5.65,4.51,4.48,5.76,8.89,10.72,11.40,11.71,11.73,11.77,11.79,12.13,12.61,13.10,13.58,13.84,13.86,13.66,13.19,12.51,10.83
],[11.61,10.73,9.53,7.80,6.22,6.14,8.09,10.88,12.19,12.93,13.35,13.45,13.68,13.90,14.17,14.62,15.05,15.32,15.35,15.19,14.81,14.22,13.52,12.32
],[12.61,11.90,10.93,8.99,6.99,7.04,9.87,12.07,13.07,13.73,14.18,14.31,14.63,14.90,15.26,15.76,16.23,16.49,16.44,16.15,15.65,14.99,14.29,13.20
],[13.38,12.79,11.96,10.64,9.38,9.64,11.59,12.74,13.54,14.10,14.57,14.71,15.11,15.47,15.93,16.49,17.00,17.25,17.15,16.78,16.21,15.54,14.88,13.90
],[13.95,13.44,12.83,11.92,11.10,11.33,12.25,13.07,13.66,14.10,14.54,14.70,15.16,15.62,16.18,16.82,17.36,17.62,17.50,17.08,16.50,15.87,15.29,14.43
],[14.34,13.89,13.38,12.76,12.16,12.12,12.57,13.13,13.48,13.75,14.14,14.30,14.80,15.36,16.03,16.75,17.33,17.60,17.48,17.07,16.52,15.97,15.49,14.77
],[14.53,14.12,13.68,13.17,12.66,12.43,12.65,12.95,13.04,13.11,13.41,13.56,14.08,14.72,15.50,16.29,16.90,17.18,17.10,16.73,16.24,15.81,15.46,14.90
],[14.48,14.14,13.75,13.31,12.83,12.52,12.53,12.57,12.40,12.26,12.43,12.55,13.06,13.78,14.65,15.48,16.08,16.38,16.35,16.06,15.66,15.35,15.15,14.78
],[14.20,13.95,13.63,13.25,12.79,12.41,12.24,12.02,11.56,11.15,11.23,11.33,11.84,12.60,13.51,14.33,14.89,15.18,15.23,15.02,14.71,14.54,14.50,14.39
],[13.66,13.54,13.32,13.00,12.58,12.14,11.79,11.34,10.67,9.87,9.90,9.92,10.36,11.19,12.09,12.80,13.32,13.56,13.67,13.54,13.32,13.29,13.42,13.69
],[12.84,12.91,12.83,12.60,12.21,11.73,11.21,10.45,9.49,8.49,8.13,8.19,8.64,9.37,10.07,10.60,10.89,11.19,11.36,11.19,10.53,10.67,11.73,12.60
],[11.34,12.03,12.14,12.04,11.72,11.19,10.40,9.50,8.25,7.18,6.86,6.81,7.15,7.68,8.25,8.34,8.09,7.89,7.96,8.02,8.03,8.55,9.35,10.25
],[8.75,10.03,11.22,11.31,11.10,10.47,9.62,6.42,7.08,6.22,5.61,5.53,5.60,5.94,5.96,5.73,5.45,5.41,5.33,5.45,5.58,5.93,6.53,8.68
],[7.21,8.43,9.01,10.42,10.27,9.66,8.70,7.30,6.17,5.20,4.49,4.42,4.27,4.34,4.39,4.17,3.97,3.60,3.52,3.56,3.74,4.22,4.84,6.02
],[5.17,6.06,7.75,9.07,9.23,8.68,7.62,6.57,5.46,4.34,3.68,3.55,3.42,3.41,3.38,2.90,2.57,2.26,2.12,2.19,2.30,2.69,3.29,4.53
],[3.94,4.57,5.53,7.23,8.03,7.82,7.02,5.80,4.42,3.56,3.11,2.89,2.78,2.47,2.24,1.96,1.60,1.28,1.22,1.17,1.29,1.55,2.03,3.06
],[2.66,3.34,4.17,5.04,5.92,6.28,5.52,4.50,3.60,2.93,2.49,2.32,2.07,1.84,1.59,1.24,0.94,0.71,0.58,0.57,0.66,0.85,1.18,2.02
],[1.68,2.30,3.04,3.87,4.37,4.51,4.24,3.64,2.90,2.31,1.92,1.85,1.57,1.36,1.07,0.81,0.54,0.34,0.20,0.22,0.26,0.36,0.61,1.18
],[1.04,1.53,2.02,2.62,3.15,3.46,3.24,2.78,2.29,1.94,1.47,1.43,1.17,0.95,0.69,0.49,0.28,0.11,0.05,0.03,0.04,0.12,0.25,0.67
],[0.59,0.92,1.32,1.73,2.06,2.34,2.31,2.02,1.70,1.39,1.14,1.06,0.84,0.66,0.46,0.20,0.12,0.03,0.00,0.00,0.00,0.02,0.06,0.34
],[0.32,0.51,0.78,1.03,1.24,1.42,1.46,1.37,1.16,1.00,0.79,0.76,0.55,0.39,0.24,0.09,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.11
],[0.17,0.24,0.43,0.60,0.72,0.80,0.84,0.85,0.72,0.65,0.51,0.48,0.35,0.22,0.09,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03
],[0.05,0.09,0.20,0.32,0.38,0.44,0.43,0.44,0.41,0.37,0.29,0.26,0.17,0.12,0.06,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.03
],[0.03,0.03,0.08,0.07,0.11,0.11,0.18,0.16,0.16,0.16,0.16,0.12,0.07,0.06,0.02,0.02,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]]
)

smartshea_2015_2 = np.array([[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.02,0.02,0.03,0.03,0.04,0.03,0.03,0.04,0.03,0.00
],[0.05,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02,0.03,0.04,0.07,0.09,0.08,0.13,0.08,0.16,0.14,0.13,0.15,0.17,0.13,0.08
],[0.31,0.12,0.06,0.02,0.00,0.00,0.00,0.02,0.04,0.08,0.15,0.18,0.28,0.34,0.32,0.35,0.41,0.38,0.42,0.47,0.47,0.46,0.43,0.39
],[0.67,0.45,0.32,0.12,0.07,0.03,0.08,0.10,0.19,0.33,0.46,0.60,0.66,0.74,0.75,0.76,0.79,0.81,0.87,0.91,0.98,1.01,0.96,0.84
],[1.28,0.97,0.65,0.39,0.27,0.19,0.20,0.35,0.54,0.78,1.00,1.13,1.30,1.35,1.39,1.39,1.44,1.47,1.56,1.72,1.76,1.86,1.80,1.60
],[2.16,1.62,1.24,0.84,0.58,0.47,0.53,0.72,1.08,1.54,1.90,2.10,2.18,2.22,2.30,2.30,2.38,2.45,2.59,2.75,2.96,3.10,2.97,2.68
],[3.34,2.72,2.07,1.48,1.07,0.92,0.99,1.34,1.93,2.64,3.16,3.40,3.49,3.49,3.53,3.58,3.69,3.81,4.05,4.30,4.61,4.72,4.54,4.02
],[4.81,4.05,3.14,2.36,1.82,1.49,1.68,2.22,3.18,4.22,4.77,4.98,5.09,5.07,5.12,5.25,5.39,5.53,5.77,6.03,6.30,6.35,5.96,5.39
],[6.33,5.38,4.60,3.54,2.73,2.42,2.53,3.46,4.77,5.87,6.87,7.33,7.26,7.11,7.08,7.32,7.67,7.99,8.34,8.93,9.30,9.22,8.63,7.65
],[8.86,7.59,5.95,4.93,3.89,3.35,3.65,4.84,6.79,8.96,9.70,9.87,9.73,9.83,10.05,10.51,10.87,11.13,11.02,11.29,11.51,11.16,10.30,9.37
],[10.34,9.57,8.40,6.57,5.19,4.40,4.77,6.74,9.61,10.95,11.55,11.73,11.77,11.79,12.13,12.61,13.10,13.58,13.84,13.86,13.66,13.19,12.51,11.59
],[12.09,11.34,10.39,8.94,7.17,5.98,6.54,9.11,11.42,12.49,13.09,13.45,13.68,13.90,14.17,14.62,15.05,15.32,15.35,15.19,14.81,14.22,13.52,12.79
],[13.00,12.39,11.62,10.40,8.22,6.77,7.57,11.04,12.46,13.32,13.90,14.31,14.63,14.90,15.26,15.76,16.23,16.49,16.44,16.15,15.65,14.99,14.29,13.62
],[13.72,13.18,12.56,11.58,10.11,9.30,10.31,12.03,13.04,13.75,14.26,14.71,15.11,15.47,15.93,16.49,17.00,17.25,17.15,16.78,16.21,15.54,14.88,14.27
],[14.27,13.79,13.27,12.57,11.59,10.99,11.65,12.54,13.29,13.81,14.24,14.70,15.16,15.62,16.18,16.82,17.36,17.62,17.50,17.08,16.50,15.87,15.29,14.76
],[14.63,14.20,13.73,13.20,12.53,12.08,12.23,12.77,13.27,13.56,13.86,14.30,14.80,15.36,16.03,16.75,17.33,17.60,17.48,17.07,16.52,15.97,15.49,15.06
],[14.78,14.39,13.98,13.52,12.99,12.54,12.47,12.77,13.00,13.05,13.19,13.56,14.08,14.72,15.50,16.29,16.90,17.18,17.10,16.73,16.24,15.81,15.46,15.14
],[14.69,14.37,14.02,13.61,13.15,12.69,12.49,12.56,12.54,12.33,12.29,12.55,13.06,13.78,14.65,15.48,16.08,16.38,16.35,16.06,15.66,15.35,15.15,14.95
],[14.33,14.12,13.85,13.51,13.10,12.65,12.33,12.19,11.90,11.41,11.16,11.33,11.84,12.60,13.51,14.33,14.89,15.18,15.23,15.02,14.71,14.54,14.50,14.47
],[13.69,13.63,13.48,13.23,12.88,12.43,12.01,11.67,11.14,10.39,9.74,9.92,10.36,11.19,12.09,12.80,13.32,13.56,13.67,13.54,13.32,13.29,13.42,13.62
],[12.71,12.88,12.90,12.77,12.49,12.06,11.54,11.05,10.13,9.13,8.30,8.19,8.64,9.37,10.07,10.60,10.89,11.19,11.36,11.19,10.53,10.67,11.73,12.32
],[10.77,11.66,12.09,12.13,11.95,11.57,10.97,10.17,9.09,7.75,6.93,6.81,7.15,7.68,8.25,8.34,8.09,7.89,7.96,8.02,8.03,8.55,9.35,9.73
],[9.09,8.84,10.64,11.30,11.29,10.96,10.18,9.23,7.94,6.61,5.99,5.53,5.60,5.94,5.96,5.73,5.45,5.41,5.33,5.45,5.58,5.93,6.53,7.75
],[6.34,7.80,8.11,9.85,10.44,10.08,9.36,8.23,6.78,5.75,4.90,4.42,4.27,4.34,4.39,4.17,3.97,3.60,3.52,3.56,3.74,4.22,4.84,5.49
],[4.52,5.40,6.64,7.92,9.36,9.11,8.39,7.13,6.09,5.01,4.07,3.55,3.42,3.41,3.38,2.90,2.57,2.26,2.12,2.19,2.30,2.69,3.29,4.07
],[3.25,4.18,4.85,6.03,7.58,8.11,7.63,6.65,5.31,4.06,3.27,2.87,2.78,2.47,2.24,1.96,1.60,1.28,1.22,1.17,1.29,1.55,2.03,2.67
],[2.18,2.80,3.75,4.50,5.30,6.20,6.08,5.19,4.19,3.32,2.70,2.32,2.07,1.84,1.59,1.24,0.94,0.71,0.58,0.57,0.66,0.85,1.18,1.59
],[1.37,1.96,2.50,3.32,3.98,4.46,4.41,4.03,3.42,2.76,2.17,1.85,1.57,1.36,1.07,0.81,0.54,0.34,0.20,0.22,0.26,0.36,0.61,0.93
],[0.77,1.16,1.67,2.28,2.80,3.33,3.50,3.21,2.60,2.16,1.71,1.43,1.17,0.95,0.69,0.49,0.28,0.11,0.05,0.03,0.04,0.12,0.25,0.49
],[0.42,0.70,1.03,1.43,1.87,2.18,2.36,2.16,1.94,1.60,1.32,1.06,0.84,0.66,0.46,0.20,0.12,0.03,0.00,0.00,0.00,0.02,0.06,0.16
],[0.21,0.34,0.59,0.85,1.13,1.37,1.46,1.41,1.32,1.13,0.91,0.76,0.55,0.39,0.24,0.09,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.05
],[0.04,0.13,0.33,0.47,0.65,0.76,0.89,0.87,0.79,0.70,0.60,0.48,0.35,0.22,0.09,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.02
],[0.02,0.06,0.17,0.21,0.31,0.39,0.45,0.43,0.42,0.39,0.33,0.26,0.17,0.12,0.06,0.03,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.03,0.03,0.08,0.07,0.11,0.11,0.18,0.16,0.16,0.16,0.12,0.12,0.07,0.06,0.02,0.02,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00
],[0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04
]])
