#!/usr/bin/env python3
"""
================================================================================
Modelo SARIMAX para previsão de PM2.5 em Bogotá
================================================================================

Proyecto Interdisciplinar - Universidade Nacional da Colômbia
Autores: Karen Daniela Marin Baez e Johann Camilo Rincon Real

Este modelo incorpora variáveis exógenas reais:
- Variáveis temporais cíclicas (hora, dia da semana, mês)
- Variáveis meteorológicas (temperatura, ENSO)
- Variáveis de emissão (incêndios florestais)

================================================================================
"""

import pandas as pd
import numpy as np
import os
import re
import json
import warnings
from zipfile import ZipFile
from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_DIR = '.'
EXOG_DIR = '.'
OUTPUT_DIR = './output'

DATA_FILES = [
    'data\dic_23-may_24.xlsx',
    'data\jun_24-nov_24.xlsx', 
    'data\dic_24-may_25.xlsx',
    'data\jun_25-nov_25.xlsx'
]

MAIN_STATION = 'Carvajal___Sevillana'

# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

def load_rmcab_file(filepath):
    """Carga archivo Excel de RMCAB."""
    df_meta = pd.read_excel(filepath, header=None, nrows=7)
    station_row = df_meta.iloc[4].tolist()
    
    col_names = ['Fecha']
    current_station = None
    for i, val in enumerate(station_row[1:], 1):
        if pd.notna(val) and val != '':
            current_station = str(val).replace(' ', '_').replace('-', '_')
            current_station = current_station.replace('á', 'a').replace('í', 'i').replace('ó', 'o')
        if current_station:
            suffix = (i - 1) % 3
            if suffix == 0:
                col_names.append(current_station)
            elif suffix == 1:
                col_names.append(f"{current_station}_NowCast")
            else:
                col_names.append(f"{current_station}_IBOCA")
    
    df = pd.read_excel(filepath, skiprows=7, header=None)
    df.columns = col_names[:len(df.columns)]
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha']).set_index('Fecha')
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_all_data(data_dir, files):
    """Carga y combina todos los archivos."""
    print("\n" + "="*60)
    print("CARREGANDO DADOS DA RMCAB")
    print("="*60)
    
    dfs = []
    for f in files:
        filepath = os.path.join(data_dir, f)
        if os.path.exists(filepath):
            print(f"  Carregando: {f}...")
            df = load_rmcab_file(filepath)
            print(f"    → {len(df)} registros")
            dfs.append(df)
    
    df_combined = pd.concat(dfs).sort_index()
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    cols_to_keep = [c for c in df_combined.columns if 'Unnamed' not in c and 'Ð"' not in c]
    df_combined = df_combined[cols_to_keep]
    
    print(f"\n✓ Total: {len(df_combined):,} registros")
    return df_combined


def load_climate_data(filepath):
    """Carga datos climáticos mensuales."""
    df = pd.read_csv(filepath, encoding='latin-1', sep=';', on_bad_lines='skip')
    
    mes_map = {'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
               'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
               'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'}
    
    df['ano_mes'] = df['Año'].astype(str) + '-' + df['Mes'].map(mes_map)
    df['temp_promedio'] = df['Temperatura promedio'].astype(str).str.replace(',', '.')
    df['temp_promedio'] = pd.to_numeric(df['temp_promedio'], errors='coerce').fillna(14.0)
    
    enos_map = {'Niña': -1, 'Neutro': 0, 'Niño': 1, 'SD': 0}
    df['ENOS_index'] = df['ENOS C'].map(enos_map).fillna(0)
    
    clima_dict = {}
    for _, row in df.iterrows():
        clima_dict[row['ano_mes']] = {
            'temperatura': row['temp_promedio'],
            'ENOS_index': row['ENOS_index']
        }
    
    return clima_dict


def load_fire_data(filepath):
    """Carga datos de incendios del KMZ."""
    incendios = {}
    
    try:
        with ZipFile(filepath, 'r') as z:
            with z.open('doc.kml') as f:
                content = f.read().decode('utf-8')
                
                pattern = r'FECHA_INCI</td>\s*<td>(\d{2}/\d{2}/\d{4})</td>'
                fechas = re.findall(pattern, content)
                
                pattern_area = r'AREA_AFECT</td>\s*<td>([\d,\.]+)</td>'
                areas = re.findall(pattern_area, content)
        
        df = pd.DataFrame({
            'fecha': pd.to_datetime(fechas, format='%d/%m/%Y', errors='coerce'),
            'area': [float(a.replace(',', '.')) for a in areas[:len(fechas)]] if areas else [0]*len(fechas)
        })
        
        df['ano_mes'] = df['fecha'].dt.strftime('%Y-%m')
        agg = df.groupby('ano_mes').agg({'fecha': 'count', 'area': 'sum'})
        
        for idx, row in agg.iterrows():
            incendios[idx] = {'n_incendios': row['fecha'], 'area_total': row['area']}
            
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    return incendios


# =============================================================================
# VARIABLES EXÓGENAS
# =============================================================================

def create_exogenous_variables(serie, clima_dict=None, fire_dict=None):
    """
    Crea todas las variables exógenas para SARIMAX.
    
    Variables:
    - hora_sin, hora_cos: Ciclo diario
    - dia_sem_sin, dia_sem_cos: Ciclo semanal
    - mes_sin, mes_cos: Ciclo anual
    - es_hora_pico: Rush matinal/vespertino
    - es_fin_de_semana: Sábado/Domingo
    - es_temporada_seca: Dic-Mar
    - temperatura: Promedio mensual
    - ENOS_index: -1/0/+1
    - incendios_mes: Número de incendios
    """
    df_exog = pd.DataFrame(index=serie.index)
    
    # Variables cíclicas
    hora = serie.index.hour
    df_exog['hora_sin'] = np.sin(2 * np.pi * hora / 24)
    df_exog['hora_cos'] = np.cos(2 * np.pi * hora / 24)
    
    dia = serie.index.dayofweek
    df_exog['dia_sem_sin'] = np.sin(2 * np.pi * dia / 7)
    df_exog['dia_sem_cos'] = np.cos(2 * np.pi * dia / 7)
    
    mes = serie.index.month
    df_exog['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    df_exog['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    
    # Variables binarias
    df_exog['es_hora_pico'] = (((hora >= 6) & (hora <= 9)) | ((hora >= 17) & (hora <= 20))).astype(int)
    df_exog['es_fin_de_semana'] = (dia >= 5).astype(int)
    df_exog['es_temporada_seca'] = serie.index.month.isin([12, 1, 2, 3]).astype(int)
    
    # Variables climáticas
    if clima_dict:
        temps, enos = [], []
        for dt in serie.index:
            key = dt.strftime('%Y-%m')
            if key in clima_dict:
                temps.append(clima_dict[key]['temperatura'])
                enos.append(clima_dict[key]['ENOS_index'])
            else:
                temps.append(14.0)
                enos.append(0)
        df_exog['temperatura'] = temps
        df_exog['ENOS_index'] = enos
    else:
        df_exog['temperatura'] = 14.0
        df_exog['ENOS_index'] = 0
    
    # Variables de incendios
    if fire_dict:
        inc = []
        for dt in serie.index:
            key = dt.strftime('%Y-%m')
            inc.append(fire_dict.get(key, {}).get('n_incendios', 0))
        df_exog['incendios_mes'] = inc
    else:
        df_exog['incendios_mes'] = 0
    
    return df_exog


def prepare_data(df, station, clima_dict, fire_dict, start_date='2023-01-01'):
    """Prepara datos y variables exógenas desde 2023."""
    serie = df[station].dropna()
    
    # Filtrar desde 2023 hasta la fecha actual
    serie = serie[start_date:]
    
    serie = serie.resample('h').mean().interpolate(method='time', limit=6).dropna()
    exog = create_exogenous_variables(serie, clima_dict, fire_dict)
    
    return serie, exog


# =============================================================================
# MODELO SARIMAX
# =============================================================================

def train_sarimax(train_y, train_exog, order=(2, 0, 1), seasonal_order=(1, 1, 1, 24)):
    """Entrena modelo SARIMAX."""
    print(f"\n  Configuración: SARIMAX{order}x{seasonal_order}")
    print(f"  Variables exógenas: {list(train_exog.columns)}")
    
    model = SARIMAX(
        train_y, exog=train_exog,
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    
    results = model.fit(disp=False, maxiter=200)
    print(f"  AIC: {results.aic:.2f}")
    
    return results


def evaluate_model(model, test_y, test_exog):
    """Evalúa el modelo."""
    pred = model.get_forecast(steps=len(test_y), exog=test_exog)
    pred_mean = pred.predicted_mean
    
    mae = mean_absolute_error(test_y, pred_mean)
    rmse = np.sqrt(mean_squared_error(test_y, pred_mean))
    
    return {
        'mae': mae, 'rmse': rmse,
        'predictions': pred_mean,
        'conf_int': pred.conf_int(alpha=0.2)
    }


def create_future_exog(last_ts, steps, clima_dict, fire_dict):
    """Crea exógenas para predicción futura."""
    future_dates = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=steps, freq='h')
    dummy = pd.Series(index=future_dates, data=0)
    return create_exogenous_variables(dummy, clima_dict, fire_dict), future_dates


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(df, station, clima_dict, fire_dict, test_hours=168):
    """Pipeline completo del modelo."""
    print("\n" + "="*60)
    print(f"MODELO SARIMAX - {station}")
    print("="*60)
    
    # Preparar datos desde 2023
    serie, exog = prepare_data(df, station, clima_dict, fire_dict, start_date='2023-01-01')
    
    print(f"\nObservaciones: {len(serie)}")
    print(f"Rango: {serie.index.min()} a {serie.index.max()}")
    print(f"Media PM2.5: {serie.mean():.1f} µg/m³")
    print(f"% crítico (>35): {(serie > 35).sum() / len(serie) * 100:.1f}%")
    
    # Dividir datos
    train_y, test_y = serie[:-test_hours], serie[-test_hours:]
    train_exog, test_exog = exog[:-test_hours], exog[-test_hours:]
    
    print(f"\nTreino: {len(train_y)} obs | Teste: {len(test_y)} obs")
    
    # Entrenar y evaluar
    model = train_sarimax(train_y, train_exog)
    metrics = evaluate_model(model, test_y, test_exog)
    
    print(f"\n  MAE: {metrics['mae']:.2f} µg/m³")
    print(f"  RMSE: {metrics['rmse']:.2f} µg/m³")
    
    # Reentrenar con todos los datos
    full_model = train_sarimax(serie, exog)
    
    # Calcular horas hasta diciembre 31, 2025 23:59
    end_forecast = pd.Timestamp('2025-12-31 23:59:59')
    last_ts = serie.index[-1]
    forecast_hours = int((end_forecast - last_ts).total_seconds() / 3600)
    
    print(f"\n  Predicción hasta: {end_forecast}")
    print(f"  Horas a predecir: {forecast_hours}")
    
    # Predicción futura hasta diciembre 2025
    future_exog, future_dates = create_future_exog(last_ts, forecast_hours, clima_dict, fire_dict)
    forecast = full_model.get_forecast(steps=forecast_hours, exog=future_exog)
    
    # Alertas
    alerts = []
    for dt, val in zip(future_dates, forecast.predicted_mean):
        if val > 35:
            alerts.append({'datetime': dt.isoformat(), 'value': round(float(val), 1), 'level': 'critical'})
        elif val > 25:
            alerts.append({'datetime': dt.isoformat(), 'value': round(float(val), 1), 'level': 'moderate'})
    
    print(f"\n  Alertas: {len(alerts)}")
    
    return {
        'station': station,
        'serie': serie,
        'exog': exog,
        'test_y': test_y,
        'test_pred': metrics['predictions'],
        'future_dates': future_dates,
        'future_mean': forecast.predicted_mean,
        'future_ci_lower': forecast.conf_int(alpha=0.2).iloc[:, 0],
        'future_ci_upper': forecast.conf_int(alpha=0.2).iloc[:, 1],
        'metrics': metrics,
        'alerts': alerts,
        'model': full_model
    }


# =============================================================================
# VISUALIZACIONES Y EXPORTACIÓN
# =============================================================================

def create_visualization(results, output_dir):
    """Crea gráfico de predicción desde 2023 hasta diciembre 2025."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = go.Figure()
    
    # Histórico completo desde 2023
    hist = results['serie']
    fig.add_trace(go.Scatter(
        x=hist.index, 
        y=hist.values, 
        name='Observado (2023-2025)', 
        line=dict(color='#1a1a1a', width=1.5),
        mode='lines'
    ))
    
    # Predicción test (última semana observada)
    fig.add_trace(go.Scatter(
        x=results['test_y'].index, 
        y=results['test_pred'].values,
        name='Predicción (test)', 
        line=dict(color='#2563eb', width=2, dash='dot')
    ))
    
    # Predicción futura hasta diciembre 2025
    fig.add_trace(go.Scatter(
        x=results['future_dates'], 
        y=results['future_mean'].values,
        name='Predicción (hasta Dic 2025)', 
        line=dict(color='#dc2626', width=2.5)
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=list(results['future_dates']) + list(results['future_dates'][::-1]),
        y=list(results['future_ci_upper'].values) + list(results['future_ci_lower'].values[::-1]),
        fill='toself', 
        fillcolor='rgba(220, 38, 38, 0.15)',
        line=dict(color='rgba(255,255,255,0)'), 
        name='IC 80%',
        showlegend=True
    ))
    
    # Líneas de referencia OMS
    fig.add_hline(y=35, line_dash="dash", line_color="red", 
                  annotation_text="Límite OMS (35 µg/m³)", 
                  annotation_position="right")
    fig.add_hline(y=15, line_dash="dash", line_color="green", 
                  annotation_text="Meta OMS (15 µg/m³)", 
                  annotation_position="right")
    
    fig.update_layout(
        title=f'Modelo SARIMAX - PM2.5 (2023-2025) | MAE: {results["metrics"]["mae"]:.1f} µg/m³ | Estación: {results["station"]}',
        xaxis_title='Fecha',
        yaxis_title='PM2.5 (µg/m³)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Mejorar visualización del eje X
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(step="all", label="Todo")
            ])
        )
    )
    
    fig.write_html(os.path.join(output_dir, 'grafico_sarimax.html'))
    print(f"\n✓ Gráfico guardado en {output_dir}/grafico_sarimax.html")


def export_data(results, clima_dict, fire_dict, output_dir):
    """Exporta datos para web."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'model': {
            'type': 'SARIMAX(2,0,1)(1,1,1,24)', 
            'mae': round(results['metrics']['mae'], 2),
            'rmse': round(results['metrics']['rmse'], 2),
            'exog_vars': list(results['exog'].columns)
        },
        'historical': {
            'dates': [d.isoformat() for d in results['serie'].index],
            'values': [round(float(v), 1) if not np.isnan(v) else None for v in results['serie'].values]
        },
        'forecast': {
            'dates': [d.isoformat() for d in results['future_dates']],
            'values': [round(float(v), 1) for v in results['future_mean'].values],
            'ci_lower': [round(float(v), 1) for v in results['future_ci_lower'].values],
            'ci_upper': [round(float(v), 1) for v in results['future_ci_upper'].values]
        },
        'alerts': results['alerts'],
        'generated_at': datetime.now().isoformat(),
        'forecast_end': results['future_dates'][-1].isoformat()
    }
    
    with open(os.path.join(output_dir, 'pm25_sarimax.json'), 'w') as f:
        json.dump(data, f, indent=2)
    
    with open(os.path.join(output_dir, 'pm25_sarimax.js'), 'w') as f:
        f.write(f"const PM25_SARIMAX = {json.dumps(data, indent=2)};")
    
    print(f"✓ Datos exportados en {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   MODELO SARIMAX CON VARIABLES EXÓGENAS (2023-2025)")
    print("="*70)
    
    # Verificar archivos
    files_found = [f for f in DATA_FILES if os.path.exists(os.path.join(DATA_DIR, f))]
    if not files_found:
        print("⚠ No se encontraron archivos RMCAB")
        return
    
    # Cargar datos
    df = load_all_data(DATA_DIR, files_found)
    
    # Cargar exógenas
    print("\n" + "="*60)
    print("CARREGANDO DADOS EXÓGENOS")
    print("="*60)
    
    clima_dict = load_climate_data(os.path.join(EXOG_DIR, 'clima.csv')) if os.path.exists(os.path.join(EXOG_DIR, 'clima.csv')) else {}
    fire_dict = load_fire_data(os.path.join(EXOG_DIR, 'incendio.kmz')) if os.path.exists(os.path.join(EXOG_DIR, 'incendio.kmz')) else {}
    
    print(f"  ✓ Clima: {len(clima_dict)} meses")
    print(f"  ✓ Incendios: {len(fire_dict)} meses")
    
    # Ejecutar modelo
    results = run_pipeline(df, MAIN_STATION, clima_dict, fire_dict)
    
    # Visualizar y exportar
    create_visualization(results, OUTPUT_DIR)
    export_data(results, clima_dict, fire_dict, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("   ¡MODELO COMPLETADO!")
    print("="*70)
    print(f"""
  Variables exógenas utilizadas:
  - Ciclos temporales: hora, día semana, mes (sin/cos)
  - Binarias: hora_pico, fin_de_semana, temporada_seca
  - Climáticas: temperatura, ENOS_index
  - Emisiones: incendios_mes

  Resultados:
  - MAE: {results['metrics']['mae']:.2f} µg/m³
  - RMSE: {results['metrics']['rmse']:.2f} µg/m³
  - Alertas: {len(results['alerts'])}
  
  Datos históricos: {results['serie'].index.min()} a {results['serie'].index.max()}
  Predicción hasta: {results['future_dates'][-1]}
""")


if __name__ == '__main__':
    main()