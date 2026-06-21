import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle
import time
import os
import zipfile
from pathlib import Path
from PIL import Image


# CORE ALGORITHM FUNCTIONS


def get_clip_hashes_with_timings(y, sr):

    timings = {}
    
    # Spectrogram
    t0 = time.time()
    stft = librosa.stft(y, n_fft=1024, hop_length=1024)
    db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    timings['spectrogram'] = int((time.time() - t0) * 1000)
    
    # Constellation (Peak picking)
    t0 = time.time()
    local_max = ndimage.maximum_filter(db_spec, size=15) == db_spec
    is_loud = db_spec > (np.max(db_spec) - 40)
    peaks = local_max & is_loud
    freq_bins, time_frames = np.where(peaks)
    
    points = list(zip(time_frames, freq_bins))
    points.sort(key=lambda x: x[0])
    timings['constellation'] = int((time.time() - t0) * 1000)
    
    # Hashing
    t0 = time.time()
    hashes = []
    FAN_VALUE = 15
    for i in range(len(points) - FAN_VALUE):
        anchor_t, anchor_f = points[i]
        for j in range(1, FAN_VALUE + 1):
            target_t, target_f = points[i + j]
            time_delta = target_t - anchor_t
            hash_key = (anchor_f, target_f, time_delta)
            hashes.append((hash_key, anchor_t))
    timings['hashing'] = int((time.time() - t0) * 1000)
    
    return hashes, time_frames, freq_bins, db_spec, timings

def find_best_match_with_timings(clip_hashes, db):
    """Finds best match, candidate list, and tracks lookup/scoring time."""
    timings = {}
    
    # DB Lookup
    t0 = time.time()
    offset_tallies = {}
    for hash_key, clip_time in clip_hashes:
        if hash_key in db:
            for db_song_name, db_time in db[hash_key]:
                offset = db_time - clip_time
                if db_song_name not in offset_tallies:
                    offset_tallies[db_song_name] = {}
                if offset not in offset_tallies[db_song_name]:
                    offset_tallies[db_song_name][offset] = 0
                offset_tallies[db_song_name][offset] += 1
    timings['lookup'] = int((time.time() - t0) * 1000)
    
    # Scoring Candidates
    t0 = time.time()
    candidate_scores = []
    for song, offsets in offset_tallies.items():
        max_score = max(offsets.values())
        best_offset = max(offsets, key=offsets.get)
        candidate_scores.append((song, max_score, best_offset))
        
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    timings['scoring'] = int((time.time() - t0) * 1000)
    
    winner = candidate_scores[0][0] if candidate_scores else "No Match Found"
    score = candidate_scores[0][1] if candidate_scores else 0
    best_offset = candidate_scores[0][2] if candidate_scores else 0
    
    return winner, score, best_offset, offset_tallies, candidate_scores, timings



# 2. APP CONFIG & STYLING


st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0b1117; color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; color: #a0aec0;
    }
    .stTabs [aria-selected="true"] { background-color: #1a202c; border-bottom: 2px solid #00ffcc; color: #00ffcc; font-weight: bold;}
    
    /* Metrics Top Bar */
    .metric-container { display: flex; justify-content: space-between; background-color: #111827; padding: 15px 25px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #1f2937; }
    .metric-box { text-align: center; }
    .metric-title { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .metric-value { font-size: 1.25rem; font-weight: bold; color: #e2e8f0; }
    .metric-sub { font-size: 0.7rem; color: #9ca3af; }
    .metric-total { text-align: right; border-left: 1px solid #374151; padding-left: 20px; }
    
    /* Match Banner */
    .match-banner { background-color: #064e3b20; border: 1px solid #047857; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
    .match-title { color: #00ffcc; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; }
    .match-song { font-size: 2.5rem; font-weight: bold; color: white; margin: 10px 0; }
    .match-stats { color: #9ca3af; font-size: 0.9rem; }
    .highlight-score { color: #f59e0b; font-weight: bold; }
    
    /* Candidate Bars */
    .candidate-row { display: flex; align-items: center; margin-bottom: 10px; font-size: 0.9rem; }
    .candidate-name { width: 300px; color: #d1d5db; }
    .candidate-bar-container { flex-grow: 1; background-color: #1f2937; height: 8px; border-radius: 4px; margin: 0 15px; overflow: hidden; }
    .candidate-bar { background-color: #00ffcc; height: 100%; border-radius: 4px; }
    .candidate-score { width: 50px; text-align: right; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

st.title("EE200: Audio Fingerprinting")
st.markdown("<p style='color: #6b7280; font-size: 0.9rem; letter-spacing: 1px;'>Made by SHASHANK KHARAYAT</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; margin-bottom: 30px;'>Identify any music clip from an indexed database using spectrogram analysis and time offset hashing.</p>", unsafe_allow_html=True)


@st.cache_resource
def load_database():
    db_filename = 'song_fingerprinting_database_EE200.pkl'
    zip_filename = 'database.zip'
    
    if not os.path.exists(db_filename) and os.path.exists(zip_filename):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()
    try:
        with open(db_filename, 'rb') as f: return pickle.load(f)
    except FileNotFoundError:
        return {}

db = load_database()

@st.cache_data
def get_db_summary(_db):
    song_stats = {}
    for hash_key, occurrences in _db.items():
        for song_name, _ in occurrences:
            song_stats[song_name] = song_stats.get(song_name, 0) + 1
    return song_stats

song_summary = get_db_summary(db)

tab1, tab2, tab3 = st.tabs(["● LIBRARY", "◎ IDENTIFY", "▦ BATCH"])

# ==========================================
# TAB 1: LIBRARY (Grid View w/ Images)
# ==========================================
with tab1:
    st.markdown("<div style='text-align: center; padding: 40px; background-color: #111827; border-radius: 8px; margin-bottom: 20px;'><h4 style='color: #9ca3af; font-weight: normal;'>50 songs in database. <br>Drop a clip in the Identify tab to test the library.</h4></div>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px;'>In the Database</h5>", unsafe_allow_html=True)
    
    if not song_summary:
        st.warning("Database is empty or could not be loaded.")
    else:
        cols_per_row = 4
        sorted_songs = sorted(song_summary.items())
        
        for i in range(0, len(sorted_songs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(sorted_songs):
                    song_name, hash_count = sorted_songs[i + j]
                    with cols[j]:
                        # Load from the Constellation_Images folder as requested
                        img_path = os.path.join("Constellation_Images", f"{song_name}.png")
                        
                        st.markdown(f"<div style='background-color: #0b1117; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #1f2937;'>", unsafe_allow_html=True)
                        if os.path.exists(img_path):
                            image = Image.open(img_path)
                            st.image(image, use_container_width=True)
                        else:
                            # Fallback if image is missing so the app doesn't break
                            fig, ax = plt.subplots(figsize=(4, 2.5), facecolor='#0b1117')
                            np.random.seed(hash(song_name) % (2**32))
                            ax.scatter(np.random.rand(150), np.random.rand(150), s=1, c='#00ffcc', alpha=0.5)
                            ax.axis('off')
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        st.markdown(f"<div style='margin-top: 10px; font-weight: bold; font-size: 0.9rem; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{song_name}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 0.75rem; color: #9ca3af;'>{hash_count:,} hashes</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: IDENTIFY (Replicating Screenshots)
# ==========================================
with tab2:
    st.markdown("<h5 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px;'>Search</h5>", unsafe_allow_html=True)
    st.markdown("### Identify a clip")
    
    uploaded_file = st.file_uploader("", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Identify", type="primary", use_container_width=False):
            with st.spinner("Analyzing audio..."):
                y, sr = librosa.load(uploaded_file, sr=22050)
                clip_frames = len(y) // 1024 # Approx length in frames
                
                # Run Algorithms
                hashes, peak_t, peak_f, db_spec, t_hash = get_clip_hashes_with_timings(y, sr)
                winner, score, best_offset, tallies, candidates, t_match = find_best_match_with_timings(hashes, db)
                
                total_time = sum(t_hash.values()) + sum(t_match.values())
                runner_up_score = candidates[1][1] if len(candidates) > 1 else 0
                
                # --- TOP METRICS BAR ---
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="metric-title">① Spectrogram</div>
                        <div class="metric-value">{t_hash['spectrogram']} ms</div>
                        <div class="metric-sub">{db_spec.shape[0]}x{db_spec.shape[1]}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">② Constellation</div>
                        <div class="metric-value">{t_hash['constellation']} ms</div>
                        <div class="metric-sub">{len(peak_t)} peaks</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">③ Hashing</div>
                        <div class="metric-value">{t_hash['hashing']} ms</div>
                        <div class="metric-sub">{len(hashes):,} hashes</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">④ DB Lookup</div>
                        <div class="metric-value">{t_match['lookup']} ms</div>
                        <div class="metric-sub">{len(candidates)} tracks</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">⑤ Scoring</div>
                        <div class="metric-value">{t_match['scoring']} ms</div>
                        <div class="metric-sub">offset {best_offset}</div>
                    </div>
                    <div class="metric-box metric-total">
                        <div class="metric-title" style="color: #00ffcc;">Total</div>
                        <div class="metric-value" style="color: #00ffcc;">{total_time} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- MATCH FOUND BANNER ---
                if score > 0:
                    st.markdown(f"""
                    <div class="match-banner">
                        <div class="match-title">Match Found</div>
                        <div class="match-song">{winner}</div>
                        <div class="match-stats">cluster score <span class="highlight-score">{score}</span> • <span class="highlight-score">{int(score/runner_up_score) if runner_up_score > 0 else '∞'}x</span> the runner-up</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("No Match Found in the database.")
                    st.stop()
                    
                # --- CANDIDATE SCORES ---
                st.markdown("<h6 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px;'>Candidate Scores</h6>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom: 40px;'>", unsafe_allow_html=True)
                for cand_name, cand_score, _ in candidates[:5]:
                    bar_width = int((cand_score / score) * 100) if score > 0 else 0
                    st.markdown(f"""
                    <div class="candidate-row">
                        <div class="candidate-name">{cand_name}</div>
                        <div class="candidate-bar-container">
                            <div class="candidate-bar" style="width: {bar_width}%;"></div>
                        </div>
                        <div class="candidate-score">{cand_score}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Helper to style plots
                def style_plot(fig, ax, xlabel, ylabel):
                    fig.patch.set_facecolor('#0b1117')
                    ax.set_facecolor('#0b1117')
                    ax.tick_params(colors='#9ca3af', labelsize=8)
                    ax.xaxis.label.set_color('#9ca3af')
                    ax.yaxis.label.set_color('#9ca3af')
                    ax.set_xlabel(xlabel, fontsize=9)
                    ax.set_ylabel(ylabel, fontsize=9)
                    for spine in ax.spines.values(): spine.set_color('#1f2937')
                
                # --- STEP 1: SPECTROGRAM TO CONSTELLATION ---
                st.markdown("<h6 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 20px;'>Step 1 • Feature Extraction</h6>", unsafe_allow_html=True)
                st.markdown("### From spectrogram to constellation")
                st.markdown(f"<p style='color: #9ca3af; font-size: 0.9rem;'>The clip was converted into a time-frequency map (left); brighter means louder at that frequency and moment. From that rich image, only the <b style='color: #00ffcc;'>{len(peak_t)} most prominent peaks</b> were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</p>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                    librosa.display.specshow(db_spec, sr=sr, hop_length=1024, x_axis='s', y_axis='hz', ax=ax1, cmap='magma')
                    style_plot(fig1, ax1, "time (s)", "frequency (Hz)")
                    st.pyplot(fig1, clear_figure=True)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                    times_sec = librosa.frames_to_time(peak_t, sr=sr, hop_length=1024)
                    freqs_hz = librosa.fft_frequencies(sr=sr, n_fft=1024)[peak_f]
                    ax2.scatter(times_sec, freqs_hz, color='#00ffcc', s=3, alpha=0.8)
                    style_plot(fig2, ax2, "time (s)", "frequency (Hz)")
                    ax2.set_ylim([0, sr/2])
                    ax2.text(0.95, 0.95, f"{len(peak_t)} peaks", transform=ax2.transAxes, ha='right', va='top', color='#00ffcc', fontsize=8)
                    st.pyplot(fig2, clear_figure=True)

                # --- STEP 2: DATABASE SEARCH (Reconstructed Full Fingerprint) ---
                st.markdown("<h6 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 40px;'>Step 2 • Database Search</h6>", unsafe_allow_html=True)
                st.markdown("### Where in the song?")
                
                # Reconstruct the full constellation of the winning song from the inverted index
                winner_points = []
                for h_key, occurrences in db.items():
                    for s_name, d_time in occurrences:
                        if s_name == winner:
                            winner_points.append((d_time, h_key[0])) # (frame_time, freq_bin)
                            
                st.markdown(f"<p style='color: #9ca3af; font-size: 0.9rem;'>The <b style='color: #00ffcc;'>{len(hashes):,} fingerprint hashes</b> were looked up against every indexed track. Below is the full fingerprint of <i>{winner}</i> reconstructed from the database, each dot is a stored hash anchor. The highlighted window is exactly where the query clip sits inside the full song.</p>", unsafe_allow_html=True)
                
                if winner_points:
                    w_times, w_freqs = zip(*winner_points)
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    ax3.scatter(w_times, w_freqs, color='#00ffcc', s=1, alpha=0.5)
                    
                    # Highlight the matched region
                    rect = patches.Rectangle((best_offset, 0), clip_frames, 1024, linewidth=1, edgecolor='#f59e0b', facecolor='#f59e0b', alpha=0.2)
                    ax3.add_patch(rect)
                    ax3.axvline(x=best_offset, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.7)
                    
                    style_plot(fig3, ax3, "time (frames)", "freq bin")
                    st.pyplot(fig3, clear_figure=True)

                # --- STEP 3: THE PROOF (Alignment Spike) ---
                st.markdown("<h6 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 40px;'>Step 3 • The Proof</h6>", unsafe_allow_html=True)
                st.markdown("### The alignment spike")
                st.markdown(f"<p style='color: #9ca3af; font-size: 0.9rem;'>Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge: <b style='color: #f59e0b;'>{score} hashes agreed on a single offset.</b> That spike cannot be a coincidence.</p>", unsafe_allow_html=True)
                
                win_tallies = tallies[winner]
                offsets = list(win_tallies.keys())
                votes = list(win_tallies.values())
                
                fig4, ax4 = plt.subplots(figsize=(10, 3.5))
                # Plot background noise
                ax4.bar(offsets, votes, width=5, color='#1f2937')
                # Highlight the spike
                ax4.bar([best_offset], [score], width=40, color='#f59e0b')
                
                ax4.annotate(f'{score} hashes\nalign here', xy=(best_offset, score), xytext=(best_offset + 300, score * 0.8),
                             arrowprops=dict(facecolor='#f59e0b', edgecolor='#f59e0b', arrowstyle='->'), color='#f59e0b', fontsize=9)
                
                ax4.text(max(offsets)*0.8, max(votes)*0.15, "chance\nmatches\n(noise floor)", color='#6b7280', fontsize=8, ha='center')
                
                style_plot(fig4, ax4, "time offset (database frame - query frame)", "# hashes")
                st.pyplot(fig4, clear_figure=True)


# ==========================================
# TAB 3: BATCH PROCESSING (CSV Strict Format)
# ==========================================
with tab3:
    st.markdown("<h5 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px;'>Batch</h5>", unsafe_allow_html=True)
    st.markdown("### Identify many clips at once")
    st.markdown("<p style='color: #9ca3af; font-size: 0.9rem;'>Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised <code>results.csv</code> with columns <code>filename, prediction</code>. The prediction is the matched track's filename without any extrension, or <code>None</code> when no candidate clears the confidence threshold.</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Query Clips", type=['wav', 'mp3'], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        if st.button("Run batch", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Identifying... {i+1}/{len(uploaded_files)}")
                try:
                    y, sr = librosa.load(file, sr=22050)
                    hashes, _, _, _, _ = get_clip_hashes_with_timings(y, sr)
                    winner, score, _, _, _, _ = find_best_match_with_timings(hashes, db)
                    
                    if winner != "No Match Found" and score > 20: # Basic confidence threshold
                        prediction_name = Path(winner).stem 
                    else:
                        prediction_name = "None"
                        
                except Exception:
                    prediction_name = "None"
                
                results.append({
                    "filename": file.name,
                    "prediction": prediction_name
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Batch complete!")
            df = pd.DataFrame(results)
            
            # Display clean results table
            st.markdown("<h6 style='color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 30px;'>Results</h6>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv",
                type="secondary"
            )
