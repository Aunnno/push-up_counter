from modules import PushupTrainer

if __name__ == "__main__":
    trainer = PushupTrainer()
    trainer.train('data/annotations.json')
    trainer.save('models/pushup_model.pkl')