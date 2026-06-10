from flask import Flask, Response, send_from_directory, request
import yfinance as yf
import pandas as pd
import numpy as np
import os, math, json

def clean(val):
    if val is None: return None
    try:
        if isinstance(val, (np.integer, np.int64, np.int32)): return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        if isinstance(val, float):
            return None if (math.isnan(val) or math.isinf(val)) else round(val, 4)
        if isinstance(val, (np.bool_)): return bool(val)
    except: return None
    return val

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 4)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_)): return bool(obj)
        return super().default(obj)

app = Flask(__name__, static_folder="static")

def safe(df, *keys):
    for key in keys:
        if key in df.index:
            return df.loc[key]
    return pd.Series([0] * len(df.columns), index=df.columns)

def avgfind(x, y): return (x + y) / 2

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/analyze")
def analyze():
    company = request.args.get("ticker", "").upper().strip()
    if not company:
        return Response(json.dumps({"error": "No ticker provided"}), mimetype='application/json'), 400

    try:
        ticker = yf.Ticker(company)
        info = ticker.info or {}

        hist = ticker.history(period="5d")
        if hist.empty:
            return Response(json.dumps({"error": f"Ticker '{company}' not found"}), mimetype='application/json'), 404

        income_df = ticker.get_income_stmt(pretty=True)[::-1]
        balance_df = ticker.get_balance_sheet(pretty=True)[::-1]

        years = [str(c)[:10] for c in income_df.columns]

        def s(df, *keys): return safe(df, *keys)
        def raw(df, key):
            if key in df.index:
                return [clean(v) for v in df.loc[key].tolist()]
            return [None] * len(years)

        # --- Liquidity ---
        current_assets      = s(balance_df, "Current Assets")
        current_liabilities = s(balance_df, "Current Liabilities")
        cash                = s(balance_df, "Cash And Cash Equivalents")
        sti                 = s(balance_df, "Other Short Term Investments")
        cash_in_hand        = cash + sti
        receivables         = s(balance_df, "Receivables")
        liquid_assets       = cash_in_hand + receivables

        current_ratio = [clean(round(a/l,3)) if l!=0 else None for a,l in zip(current_assets, current_liabilities)]
        cash_ratio    = [clean(round(c/l,3)) if l!=0 else None for c,l in zip(cash_in_hand, current_liabilities)]
        quick_ratio   = [clean(round(q/l,3)) if l!=0 else None for q,l in zip(liquid_assets, current_liabilities)]

        # --- Profitability ---
        total_revenue    = s(income_df, "Total Revenue")
        gross_profit     = s(income_df, "Gross Profit")
        operating_income = s(income_df, "Operating Income")
        net_profit       = s(income_df, "Net Income")
        assets           = s(balance_df, "Total Assets")
        cse              = s(balance_df, "Common Stock Equity")

        gpm = [clean(round((g/r)*100,2)) if r!=0 else None for g,r in zip(gross_profit, total_revenue)]
        opm = [clean(round((o/r)*100,2)) if r!=0 else None for o,r in zip(operating_income, total_revenue)]
        npm = [clean(round((n/r)*100,2)) if r!=0 else None for n,r in zip(net_profit, total_revenue)]

        roa_list, roe_list, roce_list = [], [], []
        for i in range(len(assets)-1):
            avg_a   = avgfind(assets.iloc[i], assets.iloc[i+1])
            avg_e   = avgfind(cse.iloc[i], cse.iloc[i+1])
            cap_emp = avg_a - current_liabilities.iloc[i]
            roa_list.append(clean(round((net_profit.iloc[i]/avg_a)*100,2))       if avg_a!=0   else None)
            roe_list.append(clean(round((net_profit.iloc[i]/avg_e)*100,2))       if avg_e!=0   else None)
            roce_list.append(clean(round((operating_income.iloc[i]/cap_emp)*100,2)) if cap_emp!=0 else None)

        # --- Efficiency ---
        cost_of_revenue  = s(income_df, "Cost Of Revenue")
        inventory        = s(balance_df, "Inventory")
        inventory_exists = inventory.sum() != 0
        accs    = s(balance_df, "Accounts Receivable")
        payable = s(balance_df, "Accounts Payable")

        inv_turn, dso, dpo, dsi, pay_turn, rec_turn, ast_turn = [], [], [], [], [], [], []
        for i in range(len(total_revenue)-1):
            avg_rec = avgfind(accs.iloc[i],     accs.iloc[i+1])
            avg_pay = avgfind(payable.iloc[i],  payable.iloc[i+1])
            avg_inv = avgfind(inventory.iloc[i], inventory.iloc[i+1])
            avg_ast = avgfind(assets.iloc[i],   assets.iloc[i+1])
            cor_i   = cost_of_revenue.iloc[i]
            rev_i   = total_revenue.iloc[i]

            inv_turn.append(clean(round(cor_i/avg_inv,2))          if (inventory_exists and avg_inv!=0) else None)
            dso.append(clean(round((avg_rec/rev_i)*365,2))          if rev_i!=0 else None)
            dpo.append(clean(round((avg_pay/cor_i)*365,2))          if cor_i!=0 else None)
            dsi.append(clean(round((avg_inv/cor_i)*365,2))          if (inventory_exists and cor_i!=0) else None)
            pay_turn.append(clean(round(cor_i/avg_pay,2))           if avg_pay!=0 else None)
            rec_turn.append(clean(round(rev_i/avg_rec,2))           if avg_rec!=0 else None)
            ast_turn.append(clean(round(rev_i/avg_ast,2))           if avg_ast!=0 else None)

        eff_years = years[:-1]

        # --- Leverage ---
        total_liabilities = s(balance_df, "Total Liabilities Net Minority Interest")
        interest_expense  = s(income_df, "Interest Expense")

        d2a  = [clean(round(l/a,2)) if a!=0 else None for l,a in zip(total_liabilities, assets)]
        d2e  = [clean(round(l/e,2)) if e!=0 else None for l,e in zip(total_liabilities, cse)]
        icov = [clean(round(o/i,2)) if i!=0 else None for o,i in zip(operating_income, interest_expense)]

        # --- Price ---
        eps       = s(income_df, "Diluted EPS")
        basic_eps = s(income_df, "Basic EPS")
        try:
            share_price = float(hist["Close"].iloc[-1])
        except:
            share_price = float(info.get("currentPrice") or 0)

        dps = info.get("dividendRate")

        eps_growth = ((eps.iloc[0]-eps.iloc[1])/eps.iloc[1])*100 if len(eps)>1 and eps.iloc[1]!=0 else None

        pe_ratio  = [None]*len(years)
        peg_ratio = [None]*len(years)
        if eps.iloc[0] != 0:
            pe_ratio[0] = clean(round(share_price/eps.iloc[0], 2))
        if eps_growth and eps_growth>0 and pe_ratio[0]:
            peg_ratio[0] = clean(round(pe_ratio[0]/eps_growth, 2))

        div_yield  = [None]*len(years)
        dps_series = [None]*len(years)
        div_payout = [None]*len(years)
        if dps:
            dps_series[0] = clean(dps)
            if share_price:
                div_yield[0] = clean(round((dps/share_price)*100, 2))
            div_payout = [clean(round((dps/e)*100,2)) if e and e!=0 else None for e in eps.tolist()]

        # --- Meta ---
        name       = info.get("longName", company)
        sector     = info.get("sector", "")
        industry   = info.get("industry", "")
        market_cap = info.get("marketCap")
        currency   = info.get("currency", "")
        exchange   = info.get("exchange", "")

        payload = {
            "meta": {
                "ticker": company, "name": name, "sector": sector,
                "industry": industry, "market_cap": clean(market_cap),
                "currency": currency, "exchange": exchange,
                "share_price": clean(share_price),
                "years": years,
                "eff_years": eff_years,
                "ret_years": years[:-1],
            },
            "raw": {
                "revenue":           raw(income_df, "Total Revenue"),
                "gross_profit":      raw(income_df, "Gross Profit"),
                "operating_income":  raw(income_df, "Operating Income"),
                "net_income":        raw(income_df, "Net Income"),
                "total_assets":      raw(balance_df, "Total Assets"),
                "total_liabilities": raw(balance_df, "Total Liabilities Net Minority Interest"),
                "equity":            raw(balance_df, "Common Stock Equity"),
                "ebitda":            raw(income_df, "EBITDA"),
            },
            "ratios": {
                "liquidity": {
                    "current_ratio": current_ratio,
                    "cash_ratio":    cash_ratio,
                    "quick_ratio":   quick_ratio,
                },
                "profitability": {
                    "gpm": gpm, "opm": opm, "npm": npm,
                    "roa": roa_list, "roe": roe_list, "roce": roce_list,
                },
                "efficiency": {
                    "inventory_turnover":   inv_turn,
                    "receivables_turnover": rec_turn,
                    "assets_turnover":      ast_turn,
                    "payables_turnover":    pay_turn,
                    "dso": dso, "dpo": dpo, "dsi": dsi,
                },
                "leverage": {
                    "debt_to_assets":    d2a,
                    "debt_to_equity":    d2e,
                    "interest_coverage": icov,
                },
                "price": {
                    "basic_eps":   [clean(v) for v in basic_eps.tolist()],
                    "diluted_eps": [clean(v) for v in eps.tolist()],
                    "pe_ratio":    pe_ratio,
                    "peg_ratio":   peg_ratio,
                    "div_yield":   div_yield,
                    "dps":         dps_series,
                    "div_payout":  div_payout,
                }
            }
        }

        return Response(json.dumps(payload, cls=SafeEncoder), mimetype='application/json')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), mimetype='application/json'), 500

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)