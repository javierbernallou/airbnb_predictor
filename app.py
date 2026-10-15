
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Airbnb Londres · Buscador de chollos", layout="wide")

DATA_PATH = "data/processed/listings_scored.csv.gz"
CATS = ["Infravalorado", "Precio justo", "Sobrevalorado"]
COLORS = {"Infravalorado": "#2da44e", "Precio justo": "#e3b341", "Sobrevalorado": "#d1242f"}
LEVELS = ["Básico", "Medio", "Alto", "Premium"]
MIN_N = 20      # mínimo de alojamientos para que un barrio aparezca en los rankings
MAX_POINTS = 40_000


def hex_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [175]


def demo_data(n=6000):
    rng = np.random.default_rng(1)
    bor = ["Westminster", "Camden", "Kensington and Chelsea", "Islington", "Hackney", "Tower Hamlets",
           "Southwark", "Lambeth", "Greenwich", "Croydon", "Barnet", "Ealing"]
    lat = 51.5074 + rng.normal(0, 0.06, n)
    lon = -0.1278 + rng.normal(0, 0.10, n)
    dist = np.hypot((lat - 51.5074) * 111, (lon + 0.1278) * 69.4)
    acc = rng.integers(1, 9, n)
    am = rng.integers(5, 80, n)
    fair = np.exp(3.7 + 0.2 * acc - 0.05 * dist + 0.004 * am + rng.normal(0, 0.2, n))
    return pd.DataFrame({
        "id": np.arange(n), "name": [f"Alojamiento demo {i}" for i in range(n)], "listing_url": "https://www.airbnb.com",
        "borough": rng.choice(bor, n), "room_type": rng.choice(["Entire home/apt", "Hotel room"], n, p=[.93, .07]),
        "property_type": "Entire rental unit", "latitude": lat, "longitude": lon, "accommodates": acc,
        "bedrooms": np.clip(acc // 2, 1, 5), "bathrooms": rng.choice([1, 1.5, 2], n), "amenities_count": am,
        "distance_center_km": dist, "review_scores_rating": rng.uniform(3.8, 5, n).round(2),
        "number_of_reviews": rng.integers(0, 200, n),
        "price": fair * np.exp(rng.normal(0, 0.28, n)), "pred_price": fair * np.exp(rng.normal(0, 0.10, n)),
    })


@st.cache_data(show_spinner="Cargando datos…")
def load_data():
    demo = not os.path.exists(DATA_PATH)
    df = demo_data() if demo else pd.read_csv(DATA_PATH)
    df["borough"] = df["borough"].astype(str)
    df["name"] = df["name"].fillna("(sin nombre)").astype(str)
    df["dev_pct"] = df["price"] / df["pred_price"] - 1            # < 0: más barato de lo esperado
    df["price_pp"] = df["price"] / df["accommodates"]
    df["equip_level"] = pd.qcut(df["amenities_count"].rank(method="first"), 4, labels=LEVELS).astype(str)
    return df, demo


df, is_demo = load_data()
if is_demo:
    st.warning("No encuentro `data/processed/listings_scored.csv.gz`. Estás viendo **datos sintéticos de demostración**. "
               "Ejecuta `python build_dataset.py` para usar los datos reales.")

sb = st.sidebar
sb.header("Filtros")
acc_rng = sb.slider("Nº de inquilinos", 1, 16, (1, 16))
thr = sb.slider("Umbral de precio justo (± %)", 5, 40, 15, step=5,
                help="Un alojamiento es 'justo' si su precio real está a menos de este % del predicho.") / 100
cats_sel = sb.multiselect("Valoración", CATS, default=CATS)
bor_sel = sb.multiselect("Barrio (vacío = todos)", sorted(df["borough"].unique()))
room_sel = sb.multiselect("Tipo de alojamiento", sorted(df["room_type"].unique()), default=sorted(df["room_type"].unique()))
p_lo, p_hi = int(df["price"].min()), int(df["price"].max()) + 1
price_rng = sb.slider("Precio por noche (£)", p_lo, p_hi, (p_lo, p_hi))

df["valoracion"] = np.select([df["dev_pct"] <= -thr, df["dev_pct"] >= thr], ["Infravalorado", "Sobrevalorado"], "Precio justo")
mask = (df["accommodates"].between(*acc_rng) & df["valoracion"].isin(cats_sel)
        & df["room_type"].isin(room_sel) & df["price"].between(*price_rng))
if bor_sel:
    mask &= df["borough"].isin(bor_sel)
f = df[mask]

st.title("Buscador de apartamentos por debajo de su precio estimado")
st.caption("Precio real frente al precio que el modelo considera justo según tamaño, ubicación y equipamiento.")
if f.empty:
    st.warning("Ningún alojamiento cumple estos filtros. Amplía alguno en la barra lateral.")
    st.stop()

infra, sobre = f["valoracion"] == "Infravalorado", f["valoracion"] == "Sobrevalorado"
k = st.columns(6)
k[0].metric("Alojamientos", f"{len(f):,}")
k[1].metric("Precio medio real", f"£{f['price'].mean():,.0f}")
k[2].metric("Precio medio predicho", f"£{f['pred_price'].mean():,.0f}")
k[3].metric("Infravalorados", f"{infra.mean():.1%}")
k[4].metric("Sobrevalorados", f"{sobre.mean():.1%}")
k[5].metric("Ahorro medio infravalorados", f"£{(f['pred_price'] - f['price'])[infra].mean():,.0f}" if infra.any() else "–")

t_map, t_res, t_pre, t_mod, t_cho = st.tabs(["Mapa", "Resumen", "Precios", "Modelo", "Chollos"])

with t_map:
    m = f if len(f) <= MAX_POINTS else f.sample(MAX_POINTS, random_state=1)
    if len(f) > MAX_POINTS:
        st.caption(f"Se muestran {MAX_POINTS:,} alojamientos al azar de {len(f):,} para que el mapa vaya fluido.")
    m = m.assign(color=m["valoracion"].map({c: hex_rgb(h) for c, h in COLORS.items()}),
                 real=m["price"].round(0).astype(int), pred=m["pred_price"].round(0).astype(int),
                 dev_txt=(m["dev_pct"] * 100).round(0).astype(int).astype(str) + "%")
    layer = pdk.Layer("ScatterplotLayer", data=m[["longitude", "latitude", "color", "name", "borough", "real", "pred", "dev_txt", "valoracion"]],
                      get_position="[longitude, latitude]", get_fill_color="color", get_radius=60,
                      radius_min_pixels=2, radius_max_pixels=9, pickable=True)
    tip = {"html": "<b>{name}</b><br/>{borough} · {valoracion}<br/>Real £{real} · Predicho £{pred} ({dev_txt})",
           "style": {"fontSize": "12px"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=51.51, longitude=-0.12, zoom=9.5),
                             tooltip=tip))
    st.markdown("🟢 Infravalorado &nbsp;&nbsp; 🟡 Precio justo &nbsp;&nbsp; 🔴 Sobrevalorado")

with t_res:
    g = f.groupby("borough").agg(n=("id", "size"), precio=("price", "mean"), predicho=("pred_price", "mean"),
                                 infra=("valoracion", lambda s: (s == "Infravalorado").mean())).reset_index()
    g = g[g["n"] >= MIN_N]
    a, b = st.columns(2)
    d = f["valoracion"].value_counts().reindex(CATS).fillna(0).rename_axis("valoracion").reset_index(name="n")
    a.plotly_chart(px.pie(d, names="valoracion", values="n", hole=0.55, color="valoracion", color_discrete_map=COLORS,
                          title="Reparto por valoración"))
    h = px.histogram(f[f["dev_pct"].between(-0.8, 1.5)], x="dev_pct", nbins=70, color="valoracion", color_discrete_map=COLORS,
                     category_orders={"valoracion": CATS}, title="Distribución de la desviación (real vs. predicho)")
    h.add_vline(x=-thr, line_dash="dash"); h.add_vline(x=thr, line_dash="dash")
    h.update_xaxes(tickformat=".0%", title="Precio real respecto al predicho")
    b.plotly_chart(h)
    if g.empty:
        st.info(f"Ningún barrio tiene {MIN_N} alojamientos con estos filtros.")
    else:
        c, d2 = st.columns(2)
        top = g.nlargest(15, "infra").sort_values("infra")
        c.plotly_chart(px.bar(top, x="infra", y="borough", orientation="h", title="% de infravalorados por barrio (top 15)",
                              color_discrete_sequence=[COLORS["Infravalorado"]]).update_xaxes(tickformat=".0%", title=None).update_yaxes(title=None))
        big = g.nlargest(15, "n").melt(id_vars="borough", value_vars=["precio", "predicho"], var_name="tipo", value_name="£")
        d2.plotly_chart(px.bar(big, x="borough", y="£", color="tipo", barmode="group", title="Precio medio real vs. predicho (15 barrios con más oferta)",
                               color_discrete_sequence=["#4c78a8", "#9ecae9"]).update_xaxes(title=None))
    acc = f.groupby("accommodates")["price"].median().reset_index()
    st.plotly_chart(px.bar(acc, x="accommodates", y="price", title="Precio mediano por nº de inquilinos",
                           labels={"accommodates": "Inquilinos", "price": "£ por noche"}))

with t_pre:
    orden = st.radio("Ranking de barrios por precio por persona", ["Más baratos", "Más caros"], horizontal=True)
    pp = f.groupby("borough").agg(n=("id", "size"), pp=("price_pp", "median")).reset_index()
    pp = pp[pp["n"] >= MIN_N]
    if pp.empty:
        st.info(f"Ningún barrio tiene {MIN_N} alojamientos con estos filtros.")
    else:
        sel = (pp.nsmallest(15, "pp") if orden == "Más baratos" else pp.nlargest(15, "pp")).sort_values("pp", ascending=(orden != "Más baratos"))
        st.plotly_chart(px.bar(sel, x="pp", y="borough", orientation="h", title=f"{orden}: precio mediano por persona y noche",
                               labels={"pp": "£ por persona", "borough": ""}, color_discrete_sequence=["#4c78a8"]))
    a, b = st.columns(2)
    dd = f.assign(km=f["distance_center_km"].clip(upper=25) // 1)
    dg = dd.groupby("km").agg(n=("id", "size"), Real=("price", "median"), Predicho=("pred_price", "median")).reset_index()
    dg = dg[dg["n"] >= MIN_N].melt(id_vars=["km", "n"], value_vars=["Real", "Predicho"], var_name="tipo", value_name="£")
    a.plotly_chart(px.line(dg, x="km", y="£", color="tipo", markers=True, title="Precio mediano según la distancia al centro",
                           labels={"km": "Km al centro (Charing Cross)"}, color_discrete_sequence=["#4c78a8", "#e3b341"]))
    bx = px.box(f, x="equip_level", y="price", category_orders={"equip_level": LEVELS}, points=False,
                title="Precio según nivel de equipamiento", labels={"equip_level": "Nivel (nº de amenities, por cuartiles)", "price": "£ por noche"})
    b.plotly_chart(bx.update_yaxes(type="log"))
    cols = [c for c in ["price", "pred_price", "accommodates", "bedrooms", "bathrooms", "distance_center_km", "amenities_count",
                        "review_scores_rating"] if c in f.columns]
    st.plotly_chart(px.imshow(f[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlaciones entre variables"))

with t_mod:
    err = f["price"] - f["pred_price"]
    ly, lp = np.log(f["price"]), np.log(f["pred_price"])
    r2 = 1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    k = st.columns(4)
    k[0].metric("MAE", f"£{err.abs().mean():,.1f}")
    k[1].metric("Error mediano", f"{f['dev_pct'].abs().median():.1%}")
    k[2].metric("Correlación real–predicho", f"{np.corrcoef(f['price'], f['pred_price'])[0, 1]:.3f}")
    k[3].metric("R² (log precio)", f"{r2:.3f}")
    s = f.sample(min(len(f), 6000), random_state=1)
    sc = px.scatter(s, x="pred_price", y="price", color="valoracion", color_discrete_map=COLORS, opacity=0.5, log_x=True, log_y=True,
                    category_orders={"valoracion": CATS}, hover_data=["name", "borough"], title="Precio real vs. precio predicho",
                    labels={"pred_price": "Predicho (£)", "price": "Real (£)"})
    lim = [float(min(s["price"].min(), s["pred_price"].min())), float(max(s["price"].max(), s["pred_price"].max()))]
    sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="grey", dash="dash"), name="Real = predicho"))
    st.plotly_chart(sc)
    tr = pd.qcut(f["price"], 5, labels=["Q1 (más barato)", "Q2", "Q3", "Q4", "Q5 (más caro)"], duplicates="drop")
    et = f.assign(tramo=tr, abs_err=err.abs()).groupby("tramo", observed=True)["abs_err"].mean().reset_index()
    st.plotly_chart(px.bar(et, x="tramo", y="abs_err", title="Error medio (£) por tramo de precio",
                           labels={"tramo": "", "abs_err": "MAE (£)"}, color_discrete_sequence=["#4c78a8"]))
    with st.expander("ℹCómo se calcula"):
        st.markdown("- El precio predicho viene de XGBoost entrenado con `log(precio)` y validación cruzada de 5 folds: "
                    "cada alojamiento se predice con un modelo que **no lo ha visto**.\n"
                    "- No se usan `price`, `price_pp` ni `estimated_revenue` como variables (evita fugas de información).\n"
                    "- Desviación = precio real / precio predicho − 1. Negativa: más barato de lo esperado.")

with t_cho:
    st.info("Un precio por debajo de la predicción puede ser un chollo real, o que el modelo no ve algo (fotos, estado, ruido) "
            "o que el precio del anuncio esté desactualizado. Los filtros de reseñas ayudan a descartar falsos chollos.")
    a, b, c = st.columns(3)
    min_rev = a.slider("Mínimo de reseñas", 0, 100, 10)
    min_rat = b.slider("Puntuación mínima", 3.0, 5.0, 4.5, 0.1)
    min_dev = c.slider("Mínimo % por debajo de la predicción", 10, 60, 25, 5) / 100
    ch = f[(f["dev_pct"] <= -min_dev) & (f["number_of_reviews"] >= min_rev) & (f["review_scores_rating"] >= min_rat)].sort_values("dev_pct")
    st.write(f"**{len(ch):,}** alojamientos cumplen los criterios.")
    show = ch.assign(desv=(ch["dev_pct"] * 100).round(0))[["name", "borough", "accommodates", "price", "pred_price", "desv",
                                                           "review_scores_rating", "number_of_reviews", "listing_url"]].head(200)
    st.dataframe(show, hide_index=True, column_config={
        "name": "Nombre", "borough": "Barrio", "accommodates": "Inquilinos",
        "price": st.column_config.NumberColumn("Precio £", format="%.0f"),
        "pred_price": st.column_config.NumberColumn("Predicho £", format="%.0f"),
        "desv": st.column_config.NumberColumn("Desviación %", format="%.0f"),
        "review_scores_rating": "Puntuación", "number_of_reviews": "Reseñas",
        "listing_url": st.column_config.LinkColumn("Anuncio", display_text="Abrir")})
    st.download_button("Descargar lista (CSV)", ch.to_csv(index=False).encode("utf-8"), "chollos.csv", "text/csv")