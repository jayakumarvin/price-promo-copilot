DROP TABLE IF EXISTS promo_events, fact_weekly_sales, dim_store, dim_product CASCADE;
CREATE TABLE dim_product (sku_id INT PRIMARY KEY, sku_name TEXT, category TEXT, brand TEXT,
                          base_price NUMERIC(10,2), unit_cost NUMERIC(10,2));
CREATE TABLE dim_store (store_id INT PRIMARY KEY, store_name TEXT, region TEXT, format TEXT);
CREATE TABLE fact_weekly_sales (
  week_start DATE, store_id INT REFERENCES dim_store, sku_id INT REFERENCES dim_product,
  units INT, list_price NUMERIC(10,2), net_price NUMERIC(10,2), promo_flag SMALLINT,
  promo_type TEXT, discount_pct NUMERIC(4,2), revenue NUMERIC(14,2),
  PRIMARY KEY (week_start, store_id, sku_id));
CREATE TABLE promo_events (event_id SERIAL PRIMARY KEY, week_start DATE, store_id INT, sku_id INT,
                           promo_type TEXT, discount_pct NUMERIC(4,2), trade_spend NUMERIC(14,2));
CREATE INDEX ON fact_weekly_sales (sku_id, week_start);
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'promo_ro') THEN
    CREATE ROLE promo_ro LOGIN PASSWORD 'promo_ro_pw';
  END IF;
END $$;
GRANT CONNECT ON DATABASE promo TO promo_ro;
GRANT USAGE ON SCHEMA public TO promo_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO promo_ro;
