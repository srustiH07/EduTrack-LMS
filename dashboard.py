import time

import requests
import streamlit as st


# ---------------------------------------------------------
# EduTrack LMS - Streamlit Dashboard
# ---------------------------------------------------------

# FastAPI backend URL
API_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="EduTrack LMS",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def fetch_recommendations(student_id, top_n):
    """
    Call the FastAPI recommendation endpoint and
    measure the API response time.
    """

    url = f"{API_BASE_URL}/recommendations/{student_id}"

    start_time = time.perf_counter()

    response = requests.get(
        url,
        params={"top_n": top_n},
        timeout=10
    )

    elapsed_time = time.perf_counter() - start_time

    response.raise_for_status()

    return response.json(), elapsed_time


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title("📚 EduTrack LMS")

st.subheader(
    "Adaptive Learning Path Recommendation Dashboard"
)

st.write(
    "Get personalized learning subject recommendations "
    "using collaborative filtering with a content-based "
    "cold-start fallback."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Student Selection")

student_id = st.sidebar.text_input(
    "Enter Student ID",
    value="STU-00007",
    help="Example: STU-00007 or NEW-STUDENT-001"
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# ---------------------------------------------------------
# Recommendation button
# ---------------------------------------------------------

get_recommendations = st.button(
    "🎯 Get Recommendations",
    type="primary"
)


# ---------------------------------------------------------
# Generate recommendations
# ---------------------------------------------------------

if get_recommendations:

    if not student_id.strip():

        st.error(
            "Please enter a Student ID."
        )

    else:

        try:

            # Call FastAPI
            data, response_time = fetch_recommendations(
                student_id.strip(),
                top_n
            )

            recommendations = data.get(
                "recommendations",
                []
            )

            recommendation_type = data.get(
                "recommendation_type",
                "unknown"
            )

            # -------------------------------------------------
            # Summary information
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Student ID",
                    data.get(
                        "student_id",
                        student_id
                    )
                )

            with col2:

                if recommendation_type == "hybrid":
                    display_type = "Hybrid"
                else:
                    display_type = "Content-Based Fallback"

                st.metric(
                    "Recommendation Type",
                    display_type
                )

            with col3:

                st.metric(
                    "API Response Time",
                    f"{response_time * 1000:.0f} ms"
                )

            st.divider()

            # -------------------------------------------------
            # Response-time acceptance criterion
            # -------------------------------------------------

            if response_time < 1:

                st.success(
                    f"Dashboard recommendation response completed "
                    f"in {response_time * 1000:.0f} ms — "
                    f"under the 1-second target."
                )

            else:

                st.warning(
                    f"Response time was "
                    f"{response_time * 1000:.0f} ms. "
                    f"This is above the 1-second target."
                )

            # -------------------------------------------------
            # Recommendation section
            # -------------------------------------------------

            st.header(
                "🎓 Recommended Learning Subjects"
            )

            if not recommendations:

                st.warning(
                    "No recommendations were returned "
                    "for this student."
                )

            else:

                for index, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    subject = recommendation.get(
                        "subject",
                        "Unknown Subject"
                    )

                    score = float(
                        recommendation.get(
                            "score",
                            0
                        )
                    )

                    confidence = float(
                        recommendation.get(
                            "confidence",
                            0
                        )
                    )

                    # -----------------------------------------
                    # Recommendation title
                    # -----------------------------------------

                    st.markdown(
                        f"### {index}. {subject}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Recommendation Score:** "
                            f"{score:.4f}"
                        )

                    with col2:

                        # Confidence is stored as 0-1,
                        # so display it as a percentage.
                        st.write(
                            f"**Confidence:** "
                            f"{confidence * 100:.2f}%"
                        )

                    # Streamlit progress expects
                    # a value between 0 and 1.
                    st.progress(
                        min(
                            max(
                                confidence,
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.divider()

        # -----------------------------------------------------
        # Error handling
        # -----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure FastAPI is running at "
                "http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The FastAPI server took too long to respond."
            )

        except requests.exceptions.HTTPError as error:

            st.error(
                f"FastAPI returned an error: {error}"
            )

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


# ---------------------------------------------------------
# Initial dashboard state
# ---------------------------------------------------------

else:

    st.info(
        "Enter a Student ID in the sidebar and click "
        "'Get Recommendations' to generate personalized "
        "recommendations."
    )

    st.markdown(
        "### Example Student IDs"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Existing student:**"
        )

        st.code(
            "STU-00007"
        )

    with col2:

        st.write(
            "**Cold-start student:**"
        )

        st.code(
            "NEW-STUDENT-001"
        )

    st.markdown("---")

    st.caption(
        "EduTrack LMS • Hybrid Collaborative Filtering + "
        "Content-Based Cold-Start Recommendation"
    )