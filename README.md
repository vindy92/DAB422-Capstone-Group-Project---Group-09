# DAB422-Capstone-Group-Project---Group-09
Melody Metrics -  Predicting popular songs based on audio features

### **Group 09**

### **Group members**

- Prabhleen Kaur (0857194)
- Rajwinder Kaur (0831280)
- Vindya Senadheera (0857437)
- STM Chathurangi (0850982)
- Ajay Haji Korbe (0852660)

  ### **Project Overview**

Music streaming platforms like Spotify have reshaped how people discover and engage with music. In this digital age, predicting a song's popularity is essential for artists, producers, and advertisers to stay competitive. Understanding the key drivers of a song's success enables better marketing campaigns, playlist management, and personalized recommendations.

This project explores the connection between audio features (e.g., tempo, energy, valence, and danceability) and song popularity. By analyzing a dataset of 2,409 songs with 20 audio-related features (15 numerical, 5 categorical) from the Spotify API, we aim to predict the likelihood of a song's popularity using machine learning techniques.

Through this project, we aim to provide valuable insights for artists, producers, marketers, and streaming services to enhance user experience and engagement.

### **Goals**

- Examine the relationship between audio features and song popularity.
- Develop a model to predict a song's success based on these features.

### **Dataset**

The dataset consists of 2,409 songs and 20 audio-related features extracted from the Spotify API. Key features include for the model prediction:

**Categorical Variables:**

- artist_name - Name of the artist (Nominal)

- country_name - Country where the song was produced (Nominal)

- artist_genres - List of genres associated with the artist (Nominal)

**Numerical Variables:**

- speechiness - Presence of spoken words in the track (Continuous)

- energy - Intensity and activity of the track (Continuous)

- valence - Musical positivity (Continuous)

- danceability - How suitable the track is for dancing (Continuous)

- acousticness - Confidence that the track is acoustic (Continuous)

- tempo - Tempo of the track in beats per minute (Continuous)

- liveness - Presence of a live audience (Continuous)

- duration_ms - Length of the track in milliseconds (Continuous)


**Dependent Variable:**

- track_popularity: Numerical score (0-100) indicating popularity.
  

### **Technologies Used**

- **Python:** For data processing and machine learning.

- **Flask:** Web framework for the prediction interface.

- **scikit-learn:** For building machine learning models.

- **pandas & numpy:** For data manipulation.
  
