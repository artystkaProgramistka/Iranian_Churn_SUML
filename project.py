import pandas as pd
import streamlit as st
import joblib
import os

# Panel boczny do wyboru aplikacji
st.sidebar.title("Chcesz dowiedzieć się o aplikacji?\nA może uzyskać predykcję dla rezygnacji klientów z serwisu tlekomunikacyjnego?")
selected_option = st.sidebar.radio("Wybierz Opcję", ["Informacje o Aplikacji", "Uzyskaj Predykcję"])

if selected_option == "Informacje o Aplikacji":
    st.title("Predyktor odejść klientów z serwisów telekomunikacyjnych")
    st.write("Witamy na naszej stronie!")

    # Zakładka z dodatkowymi opcjami
    tab1, tab2, tab3= st.tabs(["Informacje na Temat Aplikacji", "Przykład Działania Aplikacji", "Oceń Aplikację"])

    with tab1:
        st.subheader("Opis Aplikacji")
        st.write("Nasza aplikacja jest narzędziem stworzone dla firm bazujących na modelu subskrypcyjnym."
                 " Dzięki zaawansowanemu modelowi predykcyjnemu aplikacja pozwala zidentyfikować klientów"
                 " zagrożonych rezygnacją i zrozumieć kluczowe czynniki wpływające na ich decyzje."
                 " Uzyskaj predykcję i zwiększ retencję swoich subskrybentów.")

        tab4, tab5 = st.tabs(["Informacje o algorytmie", "informacje o danych"])

        # Informacje o algorytmie
        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**LightGBM**")
                st.write("LightGBM (Light Gradient Boosting Machine) to zaawansowany algorytm oparty na metodzie"
                         " gradientowego wzmacniania. szczególnie wydajny przy dużych zbiorach danych.")
                st.write("W naszym modelu został skonfigurowany z parametrami: ")
                st.markdown("""
                    - Pierwszy punkt
                    - Drugi punkt
                    - Trzeci punkt
                    """)

                st.write("**Strategia poszukiwania hiperparametrów:**")
                st.markdown("""
                    - Grid search z randomizacją.
                    - K-fold cross-validation (5 foldów, stratified).
                    - Max search time: bez limitu.
                    - Random seed: 1337 dla replikacji wyników.
                    """)

                st.write("**Ewaluacja i Wybór Modelu**")
                st.write("*Metody oceny wydajności modelu:*")
                st.markdown("""
                    - ROC AUC: Główna metryka oceny, osiągnięto wynik 0.980.
                    - Confusion matrix: Do oceny: 
                    """)
                st.markdown("""
                    - dokładności (Accuracy),
                    - precyzji (Precision),
                    - czułości     
                    """)

                st.write("*Polityka podziału danych:*")
                st.markdown("""
                    - Random split: Z losowym podziałem na zbiór treningowy i testowy.
                    - K-fold cross-validation: Używane do dokładniejszej ewaluacji modelu.   
                    """)

                st.write("**Wyniki modelowania:**")
                st.markdown("""
                    - Accuracy: 95%
                    - Precision: 75%
                    - Recall: 100%
                    - F1-Score: 86%  
                    """)

                # Adding the metrics visualization
                st.image("wyniki_metryki.png", caption="Metryki Jakości Modelu Uczenia Maszynowego")

                # Adding the updated explanation for model metrics in a business context
                st.markdown("""
                    ## Metryki Jakości Modelu Uczenia Maszynowego w Kontekście Biznesowym

                    Nasza aplikacja wspiera firmy działające w modelu subskrypcyjnym, pomagając identyfikować klientów zagrożonych rezygnacją. Poniżej wyjaśniamy, jak metryki modelu uczenia maszynowego pomagają w zrozumieniu i doskonaleniu strategii retencji klientów.

                    ### Definicje Metryk

                    1. **Dokładność (Accuracy):**
                       - Definicja: Odsetek poprawnych predykcji w stosunku do całkowitej liczby obserwacji.
                       - W kontekście biznesowym: Pokazuje, jak skutecznie model identyfikuje zarówno klientów, którzy zrezygnują, jak i tych, którzy pozostaną. Wysoka dokładność oznacza, że firma może precyzyjnie planować działania marketingowe lub retencyjne.

                    2. **Precyzja (Precision):**
                       - Definicja: Odsetek poprawnych predykcji pozytywnych w stosunku do wszystkich przypadków przewidzianych jako pozytywne.
                       - W kontekście biznesowym: Wysoka precyzja oznacza, że zasoby firmy (np. budżet na kampanie retencyjne) są kierowane tylko do rzeczywiście zagrożonych klientów, minimalizując koszty niepotrzebnych działań.

                    3. **Czułość (Recall):**
                       - Definicja: Odsetek poprawnych predykcji pozytywnych w stosunku do wszystkich rzeczywistych przypadków pozytywnych.
                       - W kontekście biznesowym: Wysoka czułość oznacza, że firma identyfikuje większość klientów, którzy faktycznie planują zrezygnować. Jest to kluczowe, gdy celem jest ograniczenie odpływu subskrybentów.

                    4. **F1-Score:**
                       - Definicja: Średnia harmoniczna precyzji i czułości.
                       - W kontekście biznesowym: Stanowi zrównoważoną miarę, która uwzględnia zarówno skuteczność w wykrywaniu zagrożonych klientów, jak i efektywność alokacji zasobów na kampanie retencyjne.

                    ### Podsumowanie
                    - **Dokładność:** Ogólna miara skuteczności modelu.
                    - **Precyzja:** Efektywność wykorzystania budżetu na kampanie retencyjne.
                    - **Czułość:** Zdolność identyfikacji wszystkich zagrożonych klientów.
                    - **F1-Score:** Zrównoważenie między precyzją a czułością, szczególnie istotne w kontekście kosztów i skuteczności działań retencyjnych.

                    Analiza tych metryk umożliwia firmom podejmowanie świadomych decyzji dotyczących strategii retencji, maksymalizując wartość klienta i minimalizując ryzyko rezygnacji.
                    """)

                # Inserting the feature correlation visualization image with caption
                st.image("maciez_korelacji.png",
                         caption="Macierz Korelacji Cech (Feature Correlation Matrix) modelu uczenia maszynowego")

                # Adding correlation matrix explanation
                st.markdown("""
                                ## Macierz Korelacji w Modelu Uczenia Maszynowego

                                Macierz korelacji przedstawia współzależności między różnymi cechami w danych wejściowych. Każda wartość w macierzy wskazuje siłę i kierunek związku między dwiema cechami. Analiza korelacji jest kluczowa w zrozumieniu danych i może wspierać proces optymalizacji modelu uczenia maszynowego.

                                ### Interpretacja Macierzy Korelacji
                                1. **Wartości Korelacji:**
                                   - Zakres wartości wynosi od -1 do 1:
                                     - **-1:** Silna, ujemna korelacja – gdy jedna cecha rośnie, druga maleje.
                                     - **0:** Brak korelacji – cechy są niezależne.
                                     - **1:** Silna, dodatnia korelacja – gdy jedna cecha rośnie, druga również rośnie.

                                2. **Korelacje Wysokie:**
                                   - Dodatnia korelacja (wartości bliskie 1) wskazuje na cechy, które zmieniają się w podobny sposób. Może to prowadzić do redundancji danych, co warto uwzględnić podczas trenowania modelu.

                                3. **Korelacje Niskie lub Negatywne:**
                                   - Ujemna korelacja (wartości bliskie -1) może wskazywać na cechy, które są silnie przeciwstawne. Tego typu informacje mogą być użyteczne do zrozumienia konfliktujących relacji w danych.

                                ### Zastosowanie w Kontekście Biznesowym
                                - **Redukcja Redundancji:**
                                  Jeśli dwie cechy są silnie skorelowane, jedna z nich może zostać usunięta podczas procesu inżynierii cech. To pozwala uprościć model i poprawić jego wydajność.

                                - **Zrozumienie Wpływu Cech:**
                                  Analiza korelacji pozwala firmom zidentyfikować cechy, które mają wspólny wpływ na decyzje klientów. Na przykład, częstotliwość korzystania z usługi i czas użytkowania mogą być silnie skorelowane, co wskazuje na ich podobny wkład w wynik predykcji.

                                - **Personalizacja Strategii:**
                                  Zrozumienie związków między cechami może wspierać bardziej spersonalizowane działania marketingowe. Na przykład, jeśli cechy dotyczące wieku klienta i wartości klienta są silnie skorelowane, strategia może uwzględniać specyficzne potrzeby demograficzne.

                                Macierz korelacji jest kluczowym narzędziem do eksploracji danych, umożliwiającym podejmowanie bardziej świadomych decyzji w procesie budowy i optymalizacji modeli predykcyjnych.
                                """)

            with col2:
                st.write("**Wykresy do algorytmu uczenia maszynowego:**")

                # Inserting the confusion matrix image with caption
                st.image("macierz_pomylek.png", caption="Macierz Pomyłek (Confusion Matrix) dla modelu uczenia maszynowego")

                # Adding Markdown content for Confusion Matrix explanation
                st.markdown("""
                ## Macierz Pomyłek (Confusion Matrix)
                Macierz pomyłek to narzędzie używane w uczeniu maszynowym do oceny działania modelu klasyfikacyjnego. Prezentuje ona wyniki predykcji w formie tabeli, która porównuje przewidywane klasy z rzeczywistymi klasami. Dzięki macierzy pomyłek można zrozumieć, jak dobrze model radzi sobie z klasyfikowaniem danych i gdzie popełnia błędy.

                ### Jak czytać macierz pomyłek?

                Macierz pomyłek składa się z czterech podstawowych elementów:

                1. **True Positives (TP)** - Liczba przypadków, w których model poprawnie przewidział pozytywną klasę.
                2. **True Negatives (TN)** - Liczba przypadków, w których model poprawnie przewidział negatywną klasę.
                3. **False Positives (FP)** - Liczba przypadków, w których model błędnie przewidział pozytywną klasę (tzw. fałszywy alarm).
                4. **False Negatives (FN)** - Liczba przypadków, w których model błędnie przewidział negatywną klasę (tzw. przeoczenie).

                ### Interpretacja

                - **Oś pozioma (Przewidywana wartość)**: Przedstawia klasy przewidywane przez model.
                - **Oś pionowa (Rzeczywista wartość)**: Przedstawia faktyczne klasy w danych.
                - Pola na przecięciu obu osi wskazują liczbę przypadków dla danej kombinacji przewidywań i rzeczywistości.

                Dzięki tym informacjom można obliczyć różne miary skuteczności modelu, takie jak dokładność, precyzja, czułość czy miara F1.
                """, unsafe_allow_html=False)

                # Inserting the feature importance visualization image with caption
                st.image("wykres_waznosci_cech_dla_wyniku.png", caption="Wykres Ważności Cech (Feature Importance) dla predukcji modelu uczenia maszynowego")

                # Adding feature importance explanation
                st.markdown("""
                ## Ważność Cech w Modelu Uczenia Maszynowego

                Wykres ważności cech (Feature Importance) przedstawia wpływ poszczególnych cech na decyzje podejmowane przez model uczenia maszynowego. Im wyższa wartość dla danej cechy, tym większy jej wpływ na wynik predykcji. Tego typu analiza pomaga w lepszym zrozumieniu działania modelu oraz w identyfikacji kluczowych czynników wpływających na rezygnację klientów.

                ### Interpretacja Kluczowych Cech
                1. **Subscription Length:**
                   - Ta cecha ma największy wpływ na decyzje modelu. Długość subskrypcji może sugerować lojalność klienta - im dłużej klient korzysta z usługi, tym większe prawdopodobieństwo, że pozostanie.

                2. **Age Group i Frequency of Use:**
                   - Grupa wiekowa oraz częstotliwość korzystania z usługi są również istotnymi czynnikami. Mogą wskazywać na określone grupy demograficzne lub nawyki klientów bardziej podatne na rezygnację.

                3. **Seconds of Use i Customer Value:**
                   - Czas użytkowania oraz wartość klienta dostarczają istotnych informacji o zaangażowaniu klienta w korzystanie z usługi. Niski poziom tych cech może wskazywać na klientów zagrożonych rezygnacją.

                4. **Complaints (Skargi):**
                   - Liczba zgłoszonych skarg może być bezpośrednim wskaźnikiem niezadowolenia klientów, co istotnie wpływa na ich decyzję o rezygnacji.

                ### Zastosowanie w Kontekście Biznesowym
                - **Strategie Retencji:**
                  Analiza ważności cech umożliwia firmie skupienie się na najistotniejszych aspektach, takich jak poprawa obsługi klienta (np. szybkie rozwiązywanie skarg) lub oferowanie promocji dla klientów w grupach wiekowych najbardziej podatnych na rezygnację.

                - **Personalizacja Usług:**
                  Zrozumienie, które cechy mają największe znaczenie, pozwala lepiej dopasować ofertę do potrzeb różnych segmentów klientów, zwiększając ich zadowolenie i zaangażowanie.

                - **Optymalizacja Budżetu:**
                  Dzięki identyfikacji kluczowych cech firma może efektywniej alokować zasoby, kierując kampanie retencyjne do najbardziej zagrożonych klientów.

                Wnioski płynące z wykresu ważności cech mogą być kluczowe dla podejmowania działań zwiększających retencję i minimalizujących ryzyko rezygnacji klientów.
                """)

        with tab5:

            st.markdown("""
            ### Dane Użyte w Projekcie
            Dane użyte w projekcie pochodzą z bazy danych **Iranian Churn Dataset** udostępnionej przez University of California, Irvine. Zbór danych jest dostępny pod adresem: [Iranian Churn Dataset](https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset) i objęty jest licencją **Creative Commons Attribution 4.0 International (CC BY 4.0)**, co pozwala na udostępnianie i adaptację danych do dowolnych celów, pod warunkiem udzielenia odpowiedniego uznania autorstwa.

            Dane obejmują informacje zebrane przez irańską firmę telekomunikacyjną na przestrzeni dwunastu miesięcy. Zbór danych zawiera **3150 obserwacji** oraz **13 cech**, takich jak liczba nieudanych połączeń, częstotliwość SMS-ów, rodzaj taryfy, wartość dla klienta itp. Wszystkie cechy są zagregowane z pierwszych 9 miesięcy, a etykieta rezygnacji (**Churn**) odzwierciedla stan na koniec dwunastu miesięcy. W zbiorze danych nie ma brakujących wartości.

            ---

            ### Podział Danych
            Dane zostały podzielone na zbiory:
            - **70%** do trenowania modelu,
            - **30%** do testowania modelu.

            ---

            ### Szczegółowe Informacje o Cechach Danych
            - **Call Failure** - Nieudane Połączenia *(Integer)*
            - **Frequency of SMS** - Częstotliwość wysyłania SMS-ów *(Integer)*
            - **Complaints** - Czy Złożono Skargi *(Binary)*
            - **Distinct Called Numbers** - Liczba Odrębnych Połączeń *(Integer)*
            - **Subscription Length** - Długość Abonamentu *(Integer)*
            - **Age** - Wiek *(Integer)*
            - **Age Group** - Grupa Wiekowa *(Integer)*
            - **Charge Amount** - Wysokość Opłaty *(Integer)*
            - **Tariff Plan** - Rodzaj Usługi *(Integer)*
            - **Seconds of Use** - Sekundy Użytkowania *(Integer)*
            - **Status** *(Binary)*
            - **Frequency of Use** - Częstotliwość Użytkowania *(Integer)*
            - **Customer Value** - Wartość dla Klienta *(Continuous)*

            #### Etykieta:
            - **Churn** - Czy Klient Zrezygnował z Subskrypcji *(Binary)*
            """)

    with tab2:
        df = pd.read_csv(customer_churn.csv)

        # Display a sample of the data
        st.write("Załadowane dane:")
        st.table(df.head())

        try:
            # Load the trained model
            model = joblib.load("best_rf_model.pkl")  # Ensure model file is in the same directory

            # Preprocessing the uploaded data (adjust based on `train_save_model.py`)
            df.drop(['Age'], axis=1, inplace=True, errors='ignore')  # Drop 'Age' column if it exists
            integer_columns = [col for col in df.columns if col not in ['Status', 'Complains', 'Churn']]
            df[integer_columns] = df[integer_columns].astype('int')
            df['Status'] = df['Status'].map({1: True, 2: False}).astype('bool')
            df['Complains'] = df['Complains'].map({1: True, 0: False}).astype('bool')

            # Drop the target column if it exists
            X_infer = df.drop(['Churn'], axis=1, errors='ignore')

            # Make predictions
            predictions = model.predict(X_infer)

            # Add predictions to the DataFrame
            df['Prediction'] = predictions

            # Display rows incrementally with session state tracking
            if "rows_shown" not in st.session_state:
                st.session_state.rows_shown = 50

            rows_shown = st.session_state.rows_shown
            st.write(f"Wyświetlono {rows_shown} wierszy z danymi i predykcjami:")
            st.table(df.iloc[:rows_shown])

            if rows_shown < len(df):
                if st.button(f"Pokaż kolejne {min(100, len(df) - rows_shown)} wierszy",
                             key=f"button_{rows_shown}"):
                    st.session_state.rows_shown += 100

            # Add a download button for the predictions CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Pobierz dane z predykcjami jako CSV",
                data=csv,
                file_name="predykcje.csv",
                mime="text/csv",
            )

        except FileNotFoundError:
            st.error(
                "Plik modelu `best_rf_model.pkl` nie został znaleziony. Upewnij się, że znajduje się w tym samym folderze co ten skrypt.")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas przetwarzania: {e}")

    with tab3:
        st.subheader("Oceń Aplikacje")
        st.write("Twoja opinia jest dla nas ważna! Oceń aplikację i podziel się swoimi uwagami.")
        st.write("Oceń zadowolenie aplikacji")
        rating=st.slider("Zadowolenie 0-10",0,10,5)
        feedback = st.text_area("Podziel się swoją opinią (opcjonalne):")
        if st.button("Prześlij opinię"):
            st.success("Dziękujemy za Twoją opinię!")

elif selected_option == "Uzyskaj Predykcję":
    tab6, tab7 = st.tabs(["Pojedyncza Predykcja", "Przedykcja Zbiorcza (na podstawie csv)"])

    with tab6:
        st.title("Wypełnij ręcznie pola i uzyskaj predykcję")

        try:
            # Load the trained model
            model = joblib.load("best_rf_model.pkl")

            # Input fields for user data
            st.write("Wprowadź dane klienta:")
            subscription_length = st.number_input("Długość subskrypcji (miesiące)", min_value=0, max_value=120, step=1, value=12)
            call_failures = st.number_input("Nieudane połączenia", min_value=0, max_value=100, step=1, value=5)
            complaints = st.selectbox("Czy były zgłaszane skargi?", options=["Tak", "Nie"])
            frequency_of_sms = st.number_input("Częstotliwość SMS-ów", min_value=0, max_value=500, step=1, value=50)
            customer_value = st.number_input("Wartość klienta", min_value=0.0, step=0.1, value=100.0)
            charge_amount = st.number_input("Kwota opłaty", min_value=0.0, step=0.1, value=50.0)
            tariff_plan = st.number_input("Plan taryfowy (numer)", min_value=1, max_value=10, step=1, value=1)
            distinct_called_numbers = st.number_input("Liczba odrębnych połączeń", min_value=0, max_value=100, step=1, value=10)
            frequency_of_use = st.number_input("Częstotliwość użytkowania", min_value=0, max_value=500, step=1, value=30)
            seconds_of_use = st.number_input("Czas użytkowania (sekundy)", min_value=0, max_value=100000, step=100, value=1000)
            age_group = st.number_input("Grupa wiekowa", min_value=1, max_value=5, step=1, value=2)
            status = st.selectbox("Status", options=["Aktywny", "Nieaktywny"])

            # Map input to feature names used in training
            user_data = {
                "Call  Failure": call_failures,
                "Complains": 1 if complaints == "Tak" else 0,
                "Subscription  Length": subscription_length,
                "Charge  Amount": charge_amount,
                "Seconds of Use": seconds_of_use,
                "Frequency of use": frequency_of_use,
                "Frequency of SMS": frequency_of_sms,
                "Distinct Called Numbers": distinct_called_numbers,
                "Age Group": age_group,
                "Tariff Plan": tariff_plan,
                "Status": 1 if status == "Aktywny" else 0,
                "Customer Value": customer_value,
            }

            # Convert to DataFrame for prediction
            user_df = pd.DataFrame([user_data])

            # Ensure column names match exactly with the training data
            expected_features = [
                "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
                "Seconds of Use", "Frequency of use", "Frequency of SMS",
                "Distinct Called Numbers", "Age Group", "Tariff Plan",
                "Status", "Customer Value"
            ]
            user_df = user_df[expected_features]  # Align the columns

            # Make prediction
            prediction = model.predict(user_df)
            prediction_probability = model.predict_proba(user_df)[:, 1][0]

            # Display results
            st.subheader("Wynik Predykcji:")
            if prediction[0] == 1:
                st.warning(f"Klient z dużym prawdopodobieństwem zrezygnuje. ({prediction_probability:.2%})")
            else:
                st.success(f"Klient prawdopodobnie nie zrezygnuje. ({prediction_probability:.2%})")

        except FileNotFoundError:
            st.error("Plik modelu `best_rf_model.pkl` nie został znaleziony.")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas przetwarzania: {e}")

    with tab7:
        uploaded_file = st.file_uploader("Wgraj plik .csv", type="csv")

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Display a sample of the data
            st.write("Załadowane dane:")
            st.table(df.head())

            try:
                # Load the trained model
                model = joblib.load("best_rf_model.pkl")  # Ensure model file is in the same directory

                # Preprocessing the uploaded data (adjust based on `train_save_model.py`)
                df.drop(['Age'], axis=1, inplace=True, errors='ignore')  # Drop 'Age' column if it exists
                integer_columns = [col for col in df.columns if col not in ['Status', 'Complains', 'Churn']]
                df[integer_columns] = df[integer_columns].astype('int')
                df['Status'] = df['Status'].map({1: True, 2: False}).astype('bool')
                df['Complains'] = df['Complains'].map({1: True, 0: False}).astype('bool')

                # Drop the target column if it exists
                X_infer = df.drop(['Churn'], axis=1, errors='ignore')

                # Make predictions
                predictions = model.predict(X_infer)

                # Add predictions to the DataFrame
                df['Prediction'] = predictions

                # Display rows incrementally with session state tracking
                if "rows_shown" not in st.session_state:
                    st.session_state.rows_shown = 50

                rows_shown = st.session_state.rows_shown
                st.write(f"Wyświetlono {rows_shown} wierszy z danymi i predykcjami:")
                st.table(df.iloc[:rows_shown])

                if rows_shown < len(df):
                    if st.button(f"Pokaż kolejne {min(100, len(df) - rows_shown)} wierszy",
                                 key=f"button_{rows_shown}"):
                        st.session_state.rows_shown += 100

                # Add a download button for the predictions CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Pobierz dane z predykcjami jako CSV",
                    data=csv,
                    file_name="predykcje.csv",
                    mime="text/csv",
                )

            except FileNotFoundError:
                st.error(
                    "Plik modelu `best_rf_model.pkl` nie został znaleziony. Upewnij się, że znajduje się w tym samym folderze co ten skrypt.")
            except Exception as e:
                st.error(f"Wystąpił błąd podczas przetwarzania: {e}")




